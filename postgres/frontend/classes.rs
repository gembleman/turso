//! The relation catalog: one row per user relation, in the
//! `__turso_internal_classes` table of the main database.
//!
//! `sqlite_schema` stays the source of truth for storage: type, name, root
//! page and SQL text. The catalog holds what `sqlite_schema` does not: a
//! stable `oid` that survives creating and dropping other relations, and the
//! owning role. The columns use PostgreSQL's `pg_class` names and value
//! conventions, because when `sqlite_schema` itself moves behind the
//! engine's dialect hooks, this table becomes `pg_class`, by renaming and
//! adding columns.
//!
//! Core knows nothing of this table. [`crate::catalog::PostgresDialect`]
//! maintains it through the engine's relation hooks: every statement that
//! adds, renames or removes a relation calls back here to write the matching
//! row in the same transaction as its `sqlite_schema` rows, so the two can
//! never disagree after a crash, and the engine loads the rows back as its
//! opaque dialect schema state whenever it loads the schema.
//!
//! The frontend creates the table when it opens a database, before it holds
//! any relation. A database that already has relations but no catalog was
//! written by an earlier tursopg; the frontend refuses such files instead of
//! migrating them: tursopg has made no compatibility promises yet, and a
//! refusal is better than inventing oids and owners for relations it did not
//! see created.
//!
//! Oids start at 16384, PostgreSQL's `FirstNormalObjectId`, and are never
//! reused inside one database, even after the relation is dropped: the
//! highest oid ever used is kept in `sqlite_sequence` under the catalog's
//! own name, like an AUTOINCREMENT table's watermark.

use std::any::Any;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use turso_core::namespace::NamespaceId;
use turso_core::schema::{BTreeTable, RelationKind, Schema, SQLITE_SEQUENCE_TABLE_NAME};
use turso_core::security::roles::RoleId;
use turso_core::turso_assert;
use turso_core::{
    to_u32, CmpInsFlags, CollationSeq, Connection, CursorType, IdxInsertFlags, InsertFlags, Insn,
    LimboError, Numeric, ProgramBuilder, RegisterOrLiteral, RelationChange, Value,
    SQLITE_CONSTRAINT_UNIQUE,
};
use turso_parser::ast;

use turso_core::MAIN_DB_ID;
use turso_pg_parser::translator::BOOTSTRAP_SUPERUSER_NAME;

pub(crate) const CLASSES_TABLE_NAME: &str = "__turso_internal_classes";

pub(crate) const CLASSES_NAME_INDEX_NAME: &str = "__turso_internal_classes_name";

const SELECT_CLASSES_SQL: &str =
    "SELECT oid, relname, relnamespace, relkind, relowner FROM __turso_internal_classes";

/// PostgreSQL's `FirstNormalObjectId`: the first oid given to a user
/// relation. Smaller oids belong to built-in catalog objects.
pub(crate) const FIRST_NORMAL_OBJECT_ID: i64 = 16384;

/// The columns the rest of this module reads and writes.
const CREATE_CLASSES_TABLE_SQL: &str = "CREATE TABLE __turso_internal_classes (\
    oid INTEGER PRIMARY KEY, \
    relname TEXT NOT NULL, \
    relnamespace INTEGER NOT NULL, \
    relkind TEXT NOT NULL, \
    relowner INTEGER NOT NULL)";

const CREATE_CLASSES_NAME_INDEX_SQL: &str = "CREATE UNIQUE INDEX __turso_internal_classes_name \
     ON __turso_internal_classes (relnamespace, relname)";

/// What kind of relation a class row describes, with PostgreSQL's
/// `pg_class.relkind` letters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RelKind {
    Table,
    View,
    MaterializedView,
    Sequence,
    Index,
}

impl RelKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            RelKind::Table => "r",
            RelKind::View => "v",
            RelKind::MaterializedView => "m",
            RelKind::Sequence => "S",
            RelKind::Index => "i",
        }
    }

    fn from_letter(kind: &str) -> Option<RelKind> {
        match kind {
            "r" => Some(RelKind::Table),
            "v" => Some(RelKind::View),
            "m" => Some(RelKind::MaterializedView),
            "S" => Some(RelKind::Sequence),
            "i" => Some(RelKind::Index),
            _ => None,
        }
    }
}

impl From<RelationKind> for RelKind {
    fn from(kind: RelationKind) -> RelKind {
        match kind {
            RelationKind::Table => RelKind::Table,
            RelationKind::View => RelKind::View,
            RelationKind::MaterializedView => RelKind::MaterializedView,
            RelationKind::Sequence => RelKind::Sequence,
            RelationKind::Index => RelKind::Index,
        }
    }
}

/// One row of the catalog: the identity and ownership of a relation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RelationClass {
    pub(crate) oid: i64,
    /// The relation's bare name, with the namespace in `relnamespace`.
    pub(crate) relname: String,
    pub(crate) relnamespace: NamespaceId,
    pub(crate) relkind: RelKind,
    pub(crate) relowner: RoleId,
}

/// The class rows of a database, loaded as the engine's opaque dialect
/// schema state.
#[derive(Debug, Clone, Default)]
pub(crate) struct ClassCatalog {
    by_oid: BTreeMap<i64, RelationClass>,
    oids_by_name: HashMap<(NamespaceId, String), i64>,
}

impl ClassCatalog {
    /// Returns a catalog of the rows read from the classes table. Each row is
    /// `(oid, relname, relnamespace, relkind, relowner)`.
    fn new(rows: &[Vec<Value>]) -> turso_core::Result<Self> {
        let mut catalog = Self::default();
        for row in rows {
            let class = class_from_row(row)?;
            if catalog.by_oid.contains_key(&class.oid)
                || catalog
                    .get_by_name(class.relnamespace, &class.relname)
                    .is_some()
            {
                return Err(LimboError::Corrupt(format!(
                    "{CLASSES_TABLE_NAME} has a duplicate relation: {class:?}"
                )));
            }
            catalog.add(class);
        }
        Ok(catalog)
    }

    pub(crate) fn get_by_name(&self, namespace: NamespaceId, name: &str) -> Option<&RelationClass> {
        self.oids_by_name
            .get(&(namespace, name.to_string()))
            .and_then(|oid| self.by_oid.get(oid))
    }

    /// Applies what a statement wrote to the stored rows, with `new_oid`
    /// carrying a created relation's oid.
    fn apply(&mut self, change: &RelationChange, new_oid: Option<i64>) -> turso_core::Result<()> {
        match change {
            RelationChange::Created {
                namespace,
                name,
                kind,
                owner,
            } => {
                let Some(oid) = new_oid else {
                    return Err(LimboError::InternalError(
                        "a created relation carries its oid".to_string(),
                    ));
                };
                self.add(RelationClass {
                    oid,
                    relname: name.clone(),
                    relnamespace: *namespace,
                    relkind: (*kind).into(),
                    relowner: *owner,
                });
            }
            RelationChange::Dropped { namespace, name } => {
                self.remove(*namespace, name);
            }
            RelationChange::Renamed {
                old_namespace,
                old_name,
                new_namespace,
                new_name,
            } => {
                if let Some(mut class) = self.remove(*old_namespace, old_name) {
                    class.relname.clone_from(new_name);
                    class.relnamespace = *new_namespace;
                    self.add(class);
                }
            }
            RelationChange::OwnerChanged {
                namespace,
                name,
                owner,
            } => {
                if let Some(oid) = self.oids_by_name.get(&(*namespace, name.to_string())) {
                    self.by_oid
                        .get_mut(oid)
                        .expect("both maps hold the row")
                        .relowner = *owner;
                }
            }
        }
        Ok(())
    }

    fn add(&mut self, class: RelationClass) {
        assert!(
            !self.by_oid.contains_key(&class.oid),
            "oid {} is already in use",
            class.oid
        );
        let previous = self
            .oids_by_name
            .insert((class.relnamespace, class.relname.clone()), class.oid);
        assert!(
            previous.is_none(),
            "relation {} already has a class row",
            class.relname
        );
        self.by_oid.insert(class.oid, class);
    }

    fn remove(&mut self, namespace: NamespaceId, name: &str) -> Option<RelationClass> {
        let oid = self.oids_by_name.remove(&(namespace, name.to_string()))?;
        self.by_oid.remove(&oid)
    }
}

fn class_from_row(row: &[Value]) -> turso_core::Result<RelationClass> {
    let [oid, relname, relnamespace, relkind, relowner] = row else {
        return Err(corrupt_row(row));
    };
    match (oid, relname, relnamespace, relkind, relowner) {
        (
            Value::Numeric(Numeric::Integer(oid)),
            Value::Text(relname),
            Value::Numeric(Numeric::Integer(relnamespace)),
            Value::Text(relkind),
            Value::Numeric(Numeric::Integer(relowner)),
        ) if *oid >= FIRST_NORMAL_OBJECT_ID
            && u32::try_from(*relnamespace).is_ok()
            && *relowner >= 0 =>
        {
            let Some(relkind) = RelKind::from_letter(relkind.as_str()) else {
                return Err(corrupt_row(row));
            };
            Ok(RelationClass {
                oid: *oid,
                relname: relname.as_str().to_string(),
                relnamespace: NamespaceId::new(*relnamespace as u32),
                relkind,
                relowner: RoleId::from_stored(*relowner),
            })
        }
        _ => Err(corrupt_row(row)),
    }
}

fn corrupt_row(row: &[Value]) -> LimboError {
    LimboError::Corrupt(format!("{CLASSES_TABLE_NAME} has an invalid row: {row:?}"))
}

/// The catalog that the engine loaded as this database's dialect schema
/// state, or None when it is not loaded, as in a read-only open of an empty
/// database.
pub(crate) fn class_catalog(schema: &Schema) -> Option<&ClassCatalog> {
    schema.dialect_schema.as_ref()?.downcast_ref()
}

/// [`turso_core::Dialect::schema_state_query`] for the PostgreSQL frontend.
pub(crate) fn schema_state_query(schema: &Schema) -> Option<String> {
    schema
        .get_btree_table(NamespaceId::ROOT, CLASSES_TABLE_NAME)
        .map(|_| SELECT_CLASSES_SQL.to_string())
}

/// [`turso_core::Dialect::build_schema_state`] for the PostgreSQL frontend:
/// the loaded catalog, after checking that every user relation has a row.
pub(crate) fn build_schema_state(
    schema: &Schema,
    rows: &[Vec<Value>],
) -> turso_core::Result<Arc<dyn Any + Send + Sync>> {
    let catalog = ClassCatalog::new(rows)?;
    for (namespace, name, _) in schema.user_relations() {
        turso_assert!(
            catalog.get_by_name(namespace, &name).is_some(),
            "user relation has no class row",
            { "relation": name, "namespace": namespace.as_u32() }
        );
    }
    Ok(Arc::new(catalog))
}

/// [`turso_core::Dialect::apply_relation_change`] for the PostgreSQL
/// frontend: keeps the in-memory catalog in step with the stored rows the
/// running statement wrote.
pub(crate) fn apply_relation_change(
    schema: &mut Schema,
    change: &RelationChange,
    new_oid: Option<i64>,
) -> turso_core::Result<()> {
    let Some(state) = schema.dialect_schema.take() else {
        return Ok(());
    };
    let mut catalog: Arc<ClassCatalog> = state.downcast().map_err(|_| {
        LimboError::InternalError("the dialect schema state is not the relation catalog".into())
    })?;
    Arc::make_mut(&mut catalog).apply(change, new_oid)?;
    schema.dialect_schema = Some(catalog);
    Ok(())
}

/// [`turso_core::Dialect::relation_owner`] for the PostgreSQL frontend.
pub(crate) fn relation_owner(
    schema: &Schema,
    namespace: NamespaceId,
    name: &str,
) -> Option<RoleId> {
    Some(
        class_catalog(schema)?
            .get_by_name(namespace, name)?
            .relowner,
    )
}

/// Whether a relation named `name` gets a class row. Engine objects do not;
/// the automatic `sqlite_autoindex_*` indexes of user tables do, because
/// PostgreSQL gives constraint indexes `pg_class` rows.
fn name_gets_class_row(name: &str) -> bool {
    if name.starts_with("__turso_internal_") {
        return false;
    }
    if name.starts_with("sqlite_") {
        return name.starts_with("sqlite_autoindex_");
    }
    true
}

/// [`turso_core::Dialect::emit_relations_created`] for the PostgreSQL
/// frontend: a class row per relation, owned by the current role, which is
/// what PostgreSQL's `GetUserId()` gives after `SET ROLE`.
pub(crate) fn emit_relations_created(
    program: &mut ProgramBuilder,
    schema: &Schema,
    connection: &Arc<Connection>,
    namespace: NamespaceId,
    relations: &[(String, RelationKind)],
) -> turso_core::Result<()> {
    let Some(cursors) = open_class_table_for_writing(program, schema, true)? else {
        return Ok(());
    };
    let owner = connection.current_role();
    for (name, kind) in relations {
        emit_class_row_insert(program, &cursors, namespace, name, (*kind).into(), owner);
    }
    Ok(())
}

/// [`turso_core::Dialect::emit_relations_dropped`] for the PostgreSQL
/// frontend.
pub(crate) fn emit_relations_dropped(
    program: &mut ProgramBuilder,
    schema: &Schema,
    namespace: NamespaceId,
    names: &[String],
) -> turso_core::Result<()> {
    let Some(cursors) = open_class_table_for_writing(program, schema, false)? else {
        return Ok(());
    };
    for name in names {
        emit_class_row_delete(program, &cursors, namespace, name);
    }
    Ok(())
}

/// [`turso_core::Dialect::emit_relations_renamed`] for the PostgreSQL
/// frontend. A rename keeps the relation's oid and owner.
pub(crate) fn emit_relations_renamed(
    program: &mut ProgramBuilder,
    schema: &Schema,
    namespace: NamespaceId,
    renames: &[(String, String)],
) -> turso_core::Result<()> {
    let Some(cursors) = open_class_table_for_writing(program, schema, false)? else {
        return Ok(());
    };
    for (old_name, new_name) in renames {
        emit_class_row_rename(program, &cursors, namespace, old_name, new_name);
    }
    Ok(())
}

/// [`turso_core::Dialect::emit_relation_owner_change`] for the PostgreSQL
/// frontend: `ALTER ... OWNER TO`, with PostgreSQL's errors for a missing
/// role (42704) and a catalog relation (42501).
pub(crate) fn emit_relation_owner_change(
    program: &mut ProgramBuilder,
    schema: &Schema,
    connection: &Arc<Connection>,
    namespace: NamespaceId,
    name: &str,
    new_owner: &ast::RoleSpec,
) -> turso_core::Result<()> {
    if !name_gets_class_row(name) {
        return Err(LimboError::ParseError(format!(
            "permission denied: \"{name}\" is a system catalog"
        )));
    }
    let owner = match new_owner {
        ast::RoleSpec::CurrentRole => connection.current_role(),
        ast::RoleSpec::SessionUser => connection.session_role(),
        ast::RoleSpec::Name(role_name) => {
            if role_name == BOOTSTRAP_SUPERUSER_NAME {
                RoleId::SUPERUSER
            } else {
                match schema.roles.get_by_name(role_name) {
                    Some(role) => role.id,
                    None => {
                        return Err(LimboError::ParseError(format!(
                            "role \"{role_name}\" does not exist"
                        )))
                    }
                }
            }
        }
    };
    let Some(cursors) = open_class_table_for_writing(program, schema, false)? else {
        return Err(LimboError::ParseError(
            "the database has no relation catalog".to_string(),
        ));
    };
    emit_class_row_set_owner(program, &cursors, namespace, name, owner);
    Ok(())
}

/// The cursors of one statement's class row writes: the classes table, its
/// name index, and `sqlite_sequence`, which keeps the highest oid ever used
/// under the classes table's own name.
struct ClassTableCursors {
    table_cursor_id: usize,
    index_cursor_id: usize,
    index_name: String,
    /// Only opened when the statement inserts a row, which allocates an oid.
    seq_cursor_id: Option<usize>,
    seq_table: Option<Arc<BTreeTable>>,
}

/// Opens the classes table of the main database for writing, or returns None
/// when the database has no catalog, which only happens outside the
/// frontend's own open path, such as a core-level open for tests.
/// `allocates_oids` also opens `sqlite_sequence`, which inserting a class
/// row updates.
fn open_class_table_for_writing(
    program: &mut ProgramBuilder,
    schema: &Schema,
    allocates_oids: bool,
) -> turso_core::Result<Option<ClassTableCursors>> {
    let Some(table) = schema.get_btree_table(NamespaceId::ROOT, CLASSES_TABLE_NAME) else {
        return Ok(None);
    };
    let name_index = schema
        .get_indices(NamespaceId::ROOT, CLASSES_TABLE_NAME)
        .find(|index| index.name == CLASSES_NAME_INDEX_NAME)
        .cloned()
        .ok_or_else(|| LimboError::Corrupt(format!("{CLASSES_TABLE_NAME} has no name index")))?;

    let table_cursor_id = program.alloc_cursor_id(CursorType::BTreeTable(table.clone()));
    program.emit_insn(Insn::OpenWrite {
        cursor_id: table_cursor_id,
        root_page: RegisterOrLiteral::Literal(table.root_page),
        db: MAIN_DB_ID,
    });
    let index_cursor_id = program.alloc_cursor_index(None, &name_index)?;
    program.emit_insn(Insn::OpenWrite {
        cursor_id: index_cursor_id,
        root_page: RegisterOrLiteral::Literal(name_index.root_page),
        db: MAIN_DB_ID,
    });
    let (seq_cursor_id, seq_table) = if allocates_oids {
        let Some(seq_table) = schema.get_btree_table(NamespaceId::ROOT, SQLITE_SEQUENCE_TABLE_NAME)
        else {
            return Err(LimboError::Corrupt(format!(
                "a database with {CLASSES_TABLE_NAME} has no {SQLITE_SEQUENCE_TABLE_NAME}"
            )));
        };
        let seq_cursor_id = program.alloc_cursor_id(CursorType::BTreeTable(seq_table.clone()));
        program.emit_insn(Insn::OpenWrite {
            cursor_id: seq_cursor_id,
            root_page: RegisterOrLiteral::Literal(seq_table.root_page),
            db: MAIN_DB_ID,
        });
        (Some(seq_cursor_id), Some(seq_table))
    } else {
        (None, None)
    };
    Ok(Some(ClassTableCursors {
        table_cursor_id,
        index_cursor_id,
        index_name: name_index.name.clone(),
        seq_cursor_id,
        seq_table,
    }))
}

/// Inserts the class row of relation `name` of `namespace` under the next
/// oid, owned by `owner`. Engine relations get no row.
fn emit_class_row_insert(
    program: &mut ProgramBuilder,
    cursors: &ClassTableCursors,
    namespace: NamespaceId,
    name: &str,
    kind: RelKind,
    owner: RoleId,
) {
    if !name_gets_class_row(name) {
        return;
    }
    let seq_cursor_id = cursors
        .seq_cursor_id
        .expect("the caller opened the cursors with allocates_oids");

    // oid = max(largest oid in the table + 1, highest oid ever used + 1,
    // FIRST_NORMAL_OBJECT_ID), so an oid is never reused even after the
    // relation with the largest oid is dropped.
    let (watermark_reg, watermark_rowid_reg) = emit_read_oid_watermark(program, seq_cursor_id);
    let oid_reg = program.alloc_register();
    program.emit_insn(Insn::NewRowid {
        cursor: cursors.table_cursor_id,
        rowid_reg: oid_reg,
        prev_largest_reg: 0,
    });
    let floor_reg = program.alloc_register();
    program.emit_insn(Insn::Integer {
        value: FIRST_NORMAL_OBJECT_ID - 1,
        dest: floor_reg,
    });
    program.emit_insn(Insn::MemMax {
        dest_reg: watermark_reg,
        src_reg: floor_reg,
    });
    program.emit_insn(Insn::AddImm {
        register: watermark_reg,
        value: 1,
    });
    program.emit_insn(Insn::MemMax {
        dest_reg: oid_reg,
        src_reg: watermark_reg,
    });

    emit_class_name_index_insert(program, cursors, namespace, name, oid_reg);

    // The oid is the rowid, so the record keeps NULL for it, as SQLite does
    // for an INTEGER PRIMARY KEY column.
    let first_column_reg = program.alloc_registers(5);
    program.emit_insn(Insn::Null {
        dest: first_column_reg,
        dest_end: None,
    });
    program.emit_insn(Insn::String8 {
        value: name.to_string(),
        dest: first_column_reg + 1,
    });
    program.emit_insn(Insn::Integer {
        value: i64::from(namespace.as_u32()),
        dest: first_column_reg + 2,
    });
    program.emit_insn(Insn::String8 {
        value: kind.as_str().to_string(),
        dest: first_column_reg + 3,
    });
    program.emit_insn(Insn::Integer {
        value: owner.get(),
        dest: first_column_reg + 4,
    });
    let record_reg = program.alloc_register();
    program.emit_insn(Insn::MakeRecord {
        start_reg: to_u32(first_column_reg),
        count: to_u32(5),
        dest_reg: to_u32(record_reg),
        index_name: None,
        affinity_str: None,
    });
    program.emit_insn(Insn::Insert {
        cursor: cursors.table_cursor_id,
        key_reg: oid_reg,
        record_reg,
        flag: InsertFlags::new()
            .skip_last_rowid()
            .skip_all_change_counts(),
        table_name: CLASSES_TABLE_NAME.to_string(),
    });

    emit_write_oid_watermark(
        program,
        cursors,
        seq_cursor_id,
        watermark_rowid_reg,
        oid_reg,
    );

    let kind = match kind {
        RelKind::Table => RelationKind::Table,
        RelKind::View => RelationKind::View,
        RelKind::MaterializedView => RelationKind::MaterializedView,
        RelKind::Sequence => RelationKind::Sequence,
        RelKind::Index => RelationKind::Index,
    };
    program.emit_insn(Insn::DialectRelationChange {
        db: MAIN_DB_ID,
        change: Box::new(RelationChange::Created {
            namespace,
            name: name.to_string(),
            kind,
            owner,
        }),
        id_reg: Some(oid_reg),
    });
}

/// Deletes the class row of relation `name` of `namespace`, when it has one.
fn emit_class_row_delete(
    program: &mut ProgramBuilder,
    cursors: &ClassTableCursors,
    namespace: NamespaceId,
    name: &str,
) {
    if !name_gets_class_row(name) {
        return;
    }
    emit_find_class_row(program, cursors, namespace, name, |program, row| {
        emit_class_name_index_delete(program, cursors, namespace, name, row.rowid_reg);
        // Marked as part of an update so that changes() does not count it.
        program.emit_insn(Insn::Delete {
            cursor_id: cursors.table_cursor_id,
            table_name: CLASSES_TABLE_NAME.to_string(),
            is_part_of_update: true,
        });
    });
    program.emit_insn(Insn::DialectRelationChange {
        db: MAIN_DB_ID,
        change: Box::new(RelationChange::Dropped {
            namespace,
            name: name.to_string(),
        }),
        id_reg: None,
    });
}

/// Renames the class row of relation `old_name` of `namespace` to
/// `new_name`, keeping its oid and owner.
fn emit_class_row_rename(
    program: &mut ProgramBuilder,
    cursors: &ClassTableCursors,
    namespace: NamespaceId,
    old_name: &str,
    new_name: &str,
) {
    if !name_gets_class_row(old_name) {
        return;
    }
    emit_find_class_row(program, cursors, namespace, old_name, |program, row| {
        emit_class_name_index_delete(program, cursors, namespace, old_name, row.rowid_reg);
        emit_class_name_index_insert(program, cursors, namespace, new_name, row.rowid_reg);
        let record_reg = emit_class_record(
            program,
            ClassRecordColumn::Literal(new_name.to_string()),
            ClassRecordColumn::LiteralInt(i64::from(namespace.as_u32())),
            ClassRecordColumn::Register(row.relkind_reg),
            ClassRecordColumn::Register(row.relowner_reg),
        );
        emit_class_row_overwrite(program, cursors, row.rowid_reg, record_reg);
    });
    program.emit_insn(Insn::DialectRelationChange {
        db: MAIN_DB_ID,
        change: Box::new(RelationChange::Renamed {
            old_namespace: namespace,
            old_name: old_name.to_string(),
            new_namespace: namespace,
            new_name: new_name.to_string(),
        }),
        id_reg: None,
    });
}

/// Sets the owner of the class row of relation `name` of `namespace`.
fn emit_class_row_set_owner(
    program: &mut ProgramBuilder,
    cursors: &ClassTableCursors,
    namespace: NamespaceId,
    name: &str,
    owner: RoleId,
) {
    emit_find_class_row(program, cursors, namespace, name, |program, row| {
        let record_reg = emit_class_record(
            program,
            ClassRecordColumn::Literal(name.to_string()),
            ClassRecordColumn::LiteralInt(i64::from(namespace.as_u32())),
            ClassRecordColumn::Register(row.relkind_reg),
            ClassRecordColumn::LiteralInt(owner.get()),
        );
        emit_class_row_overwrite(program, cursors, row.rowid_reg, record_reg);
    });
    program.emit_insn(Insn::DialectRelationChange {
        db: MAIN_DB_ID,
        change: Box::new(RelationChange::OwnerChanged {
            namespace,
            name: name.to_string(),
            owner,
        }),
        id_reg: None,
    });
}

/// The registers that [`emit_find_class_row`] fills for the found row.
struct FoundClassRow {
    rowid_reg: usize,
    relkind_reg: usize,
    relowner_reg: usize,
}

/// Scans the classes table for the row of relation `name` of `namespace` and
/// runs `found` with the cursor on it. Emits nothing after the scan when the
/// row does not exist, so a relation without a row is tolerated.
fn emit_find_class_row(
    program: &mut ProgramBuilder,
    cursors: &ClassTableCursors,
    namespace: NamespaceId,
    name: &str,
    found: impl FnOnce(&mut ProgramBuilder, &FoundClassRow),
) {
    let name_reg = program.emit_string8_new_reg(name.to_string());
    let namespace_reg = program.alloc_register();
    program.emit_insn(Insn::Integer {
        value: i64::from(namespace.as_u32()),
        dest: namespace_reg,
    });

    let loop_end = program.allocate_label();
    let done = program.allocate_label();
    program.emit_insn(Insn::Rewind {
        cursor_id: cursors.table_cursor_id,
        pc_if_empty: loop_end,
    });
    let loop_start = program.allocate_label();
    program.preassign_label_to_next_insn(loop_start);

    let next = program.allocate_label();
    let column_reg = program.alloc_register();
    program.emit_column_or_rowid(cursors.table_cursor_id, 2, column_reg);
    program.emit_insn(Insn::Ne {
        lhs: column_reg,
        rhs: namespace_reg,
        target_pc: next,
        flags: CmpInsFlags::default().jump_if_null(),
        collation: CollationSeq::for_internal_values(),
    });
    program.emit_column_or_rowid(cursors.table_cursor_id, 1, column_reg);
    program.emit_insn(Insn::Ne {
        lhs: column_reg,
        rhs: name_reg,
        target_pc: next,
        flags: CmpInsFlags::default().jump_if_null(),
        collation: CollationSeq::NOCASE,
    });

    let row = FoundClassRow {
        rowid_reg: program.alloc_register(),
        relkind_reg: program.alloc_register(),
        relowner_reg: program.alloc_register(),
    };
    program.emit_insn(Insn::RowId {
        cursor_id: cursors.table_cursor_id,
        dest: row.rowid_reg,
    });
    program.emit_column_or_rowid(cursors.table_cursor_id, 3, row.relkind_reg);
    program.emit_column_or_rowid(cursors.table_cursor_id, 4, row.relowner_reg);
    found(program, &row);
    program.emit_insn(Insn::Goto { target_pc: done });

    program.preassign_label_to_next_insn(next);
    program.emit_insn(Insn::Next {
        cursor_id: cursors.table_cursor_id,
        pc_if_next: loop_start,
        fullscan: false,
        is_index: false,
    });
    program.preassign_label_to_next_insn(loop_end);
    program.preassign_label_to_next_insn(done);
}

/// What a column of a rewritten class record is built from.
enum ClassRecordColumn {
    Literal(String),
    LiteralInt(i64),
    Register(usize),
}

/// Builds a class record `(NULL, relname, relnamespace, relkind, relowner)`
/// and returns its register.
fn emit_class_record(
    program: &mut ProgramBuilder,
    relname: ClassRecordColumn,
    relnamespace: ClassRecordColumn,
    relkind: ClassRecordColumn,
    relowner: ClassRecordColumn,
) -> usize {
    let first_column_reg = program.alloc_registers(5);
    program.emit_insn(Insn::Null {
        dest: first_column_reg,
        dest_end: None,
    });
    for (offset, column) in [relname, relnamespace, relkind, relowner]
        .into_iter()
        .enumerate()
    {
        let dest = first_column_reg + 1 + offset;
        match column {
            ClassRecordColumn::Literal(value) => {
                program.emit_insn(Insn::String8 { value, dest });
            }
            ClassRecordColumn::LiteralInt(value) => {
                program.emit_insn(Insn::Integer { value, dest });
            }
            ClassRecordColumn::Register(src_reg) => {
                program.emit_insn(Insn::Copy {
                    src_reg,
                    dst_reg: dest,
                    extra_amount: 0,
                });
            }
        }
    }
    let record_reg = program.alloc_register();
    program.emit_insn(Insn::MakeRecord {
        start_reg: to_u32(first_column_reg),
        count: to_u32(5),
        dest_reg: to_u32(record_reg),
        index_name: None,
        affinity_str: None,
    });
    record_reg
}

/// Replaces the class row at `rowid_reg` with `record_reg`. The cursor is on
/// the row. The delete first ends the old version under MVCC, where an
/// update is a delete and an insert.
fn emit_class_row_overwrite(
    program: &mut ProgramBuilder,
    cursors: &ClassTableCursors,
    rowid_reg: usize,
    record_reg: usize,
) {
    if program.is_mvcc_enabled() {
        program.emit_insn(Insn::Delete {
            cursor_id: cursors.table_cursor_id,
            table_name: CLASSES_TABLE_NAME.to_string(),
            is_part_of_update: true,
        });
    }
    program.emit_insn(Insn::Insert {
        cursor: cursors.table_cursor_id,
        key_reg: rowid_reg,
        record_reg,
        flag: InsertFlags::new()
            .skip_last_rowid()
            .skip_all_change_counts()
            .require_seek(),
        table_name: CLASSES_TABLE_NAME.to_string(),
    });
}

/// Adds `(namespace, name, rowid)` to the name index, failing if another
/// relation of the namespace has the name. The schema has already rejected a
/// name it knows, so this only fails if the stored table has a relation the
/// schema does not.
fn emit_class_name_index_insert(
    program: &mut ProgramBuilder,
    cursors: &ClassTableCursors,
    namespace: NamespaceId,
    name: &str,
    rowid_reg: usize,
) {
    let key_reg = program.alloc_registers(3);
    program.emit_insn(Insn::Integer {
        value: i64::from(namespace.as_u32()),
        dest: key_reg,
    });
    program.emit_insn(Insn::String8 {
        value: name.to_string(),
        dest: key_reg + 1,
    });
    program.emit_insn(Insn::Copy {
        src_reg: rowid_reg,
        dst_reg: key_reg + 2,
        extra_amount: 0,
    });
    let no_conflict = program.allocate_label();
    program.emit_insn(Insn::NoConflict {
        cursor_id: cursors.index_cursor_id,
        target_pc: no_conflict,
        record_reg: key_reg,
        num_regs: 2,
    });
    program.emit_insn(Insn::Halt {
        constraint_info: None,
        err_code: SQLITE_CONSTRAINT_UNIQUE,
        description: format!("relation \"{name}\" already has a class row"),
        on_error: None,
        description_reg: None,
    });
    program.preassign_label_to_next_insn(no_conflict);
    let record_reg = program.alloc_register();
    program.emit_insn(Insn::MakeRecord {
        start_reg: to_u32(key_reg),
        count: to_u32(3),
        dest_reg: to_u32(record_reg),
        index_name: Some(cursors.index_name.clone()),
        affinity_str: None,
    });
    program.emit_insn(Insn::IdxInsert {
        cursor_id: cursors.index_cursor_id,
        record_reg,
        unpacked_start: Some(key_reg),
        unpacked_count: Some(3),
        flags: IdxInsertFlags::new(),
    });
}

/// Removes `(namespace, name, rowid)` from the name index.
fn emit_class_name_index_delete(
    program: &mut ProgramBuilder,
    cursors: &ClassTableCursors,
    namespace: NamespaceId,
    name: &str,
    rowid_reg: usize,
) {
    let key_reg = program.alloc_registers(3);
    program.emit_insn(Insn::Integer {
        value: i64::from(namespace.as_u32()),
        dest: key_reg,
    });
    program.emit_insn(Insn::String8 {
        value: name.to_string(),
        dest: key_reg + 1,
    });
    program.emit_insn(Insn::Copy {
        src_reg: rowid_reg,
        dst_reg: key_reg + 2,
        extra_amount: 0,
    });
    program.emit_insn(Insn::IdxDelete {
        start_reg: key_reg,
        num_regs: 3,
        cursor_id: cursors.index_cursor_id,
        raise_error_if_no_matching_entry: true,
    });
}

/// Reads the highest oid ever used from the `sqlite_sequence` row named
/// after the classes table. Returns the value register, an integer, zero
/// when the row does not exist, and the register of the row's rowid, NULL
/// when it does not exist.
fn emit_read_oid_watermark(program: &mut ProgramBuilder, seq_cursor_id: usize) -> (usize, usize) {
    let watermark_reg = program.alloc_register();
    let watermark_rowid_reg = program.alloc_register();
    program.emit_insn(Insn::Integer {
        value: 0,
        dest: watermark_reg,
    });
    program.emit_insn(Insn::Null {
        dest: watermark_rowid_reg,
        dest_end: None,
    });
    let name_reg = program.emit_string8_new_reg(CLASSES_TABLE_NAME.to_string());

    let loop_end = program.allocate_label();
    program.emit_insn(Insn::Rewind {
        cursor_id: seq_cursor_id,
        pc_if_empty: loop_end,
    });
    let loop_start = program.allocate_label();
    program.preassign_label_to_next_insn(loop_start);

    let next = program.allocate_label();
    let name_column_reg = program.alloc_register();
    program.emit_column_or_rowid(seq_cursor_id, 0, name_column_reg);
    program.emit_insn(Insn::Ne {
        lhs: name_column_reg,
        rhs: name_reg,
        target_pc: next,
        flags: CmpInsFlags::default().jump_if_null(),
        collation: CollationSeq::NOCASE,
    });
    program.emit_column_or_rowid(seq_cursor_id, 1, watermark_reg);
    // AddImm leaves an integer even when the stored value is not one.
    program.emit_insn(Insn::AddImm {
        register: watermark_reg,
        value: 0,
    });
    program.emit_insn(Insn::RowId {
        cursor_id: seq_cursor_id,
        dest: watermark_rowid_reg,
    });
    program.emit_insn(Insn::Goto {
        target_pc: loop_end,
    });

    program.preassign_label_to_next_insn(next);
    program.emit_insn(Insn::Next {
        cursor_id: seq_cursor_id,
        pc_if_next: loop_start,
        fullscan: false,
        is_index: false,
    });
    program.preassign_label_to_next_insn(loop_end);
    (watermark_reg, watermark_rowid_reg)
}

/// Writes `oid_reg` as the highest oid ever used to the `sqlite_sequence`
/// row whose rowid [`emit_read_oid_watermark`] put in `watermark_rowid_reg`,
/// inserting the row when there is none.
fn emit_write_oid_watermark(
    program: &mut ProgramBuilder,
    cursors: &ClassTableCursors,
    seq_cursor_id: usize,
    watermark_rowid_reg: usize,
    oid_reg: usize,
) {
    let record_start_reg = program.alloc_registers(2);
    program.emit_insn(Insn::String8 {
        value: CLASSES_TABLE_NAME.to_string(),
        dest: record_start_reg,
    });
    program.emit_insn(Insn::Copy {
        src_reg: oid_reg,
        dst_reg: record_start_reg + 1,
        extra_amount: 0,
    });
    let affinity_str = cursors
        .seq_table
        .as_ref()
        .expect("the caller opened the cursors with allocates_oids")
        .columns()
        .iter()
        .map(|col| col.affinity().aff_mask())
        .collect::<String>();
    let record_reg = program.alloc_register();
    program.emit_insn(Insn::MakeRecord {
        start_reg: to_u32(record_start_reg),
        count: to_u32(2),
        dest_reg: to_u32(record_reg),
        index_name: None,
        affinity_str: Some(affinity_str),
    });

    let update_existing = program.allocate_label();
    let insert_new = program.allocate_label();
    let done = program.allocate_label();
    program.emit_insn(Insn::NotNull {
        reg: watermark_rowid_reg,
        target_pc: update_existing,
    });
    program.preassign_label_to_next_insn(insert_new);
    let new_rowid_reg = program.alloc_register();
    program.emit_insn(Insn::NewRowid {
        cursor: seq_cursor_id,
        rowid_reg: new_rowid_reg,
        prev_largest_reg: 0,
    });
    program.emit_insn(Insn::Insert {
        cursor: seq_cursor_id,
        key_reg: new_rowid_reg,
        record_reg,
        flag: InsertFlags::new()
            .skip_last_rowid()
            .skip_all_change_counts(),
        table_name: SQLITE_SEQUENCE_TABLE_NAME.to_string(),
    });
    program.emit_insn(Insn::Goto { target_pc: done });

    program.preassign_label_to_next_insn(update_existing);
    // The row may have vanished since it was found, e.g. when a statement
    // inserts several class rows and another write moved it.
    program.emit_insn(Insn::NotExists {
        cursor: seq_cursor_id,
        rowid_reg: watermark_rowid_reg,
        target_pc: insert_new,
    });
    if program.is_mvcc_enabled() {
        program.emit_insn(Insn::Delete {
            cursor_id: seq_cursor_id,
            table_name: SQLITE_SEQUENCE_TABLE_NAME.to_string(),
            is_part_of_update: true,
        });
    }
    program.emit_insn(Insn::Insert {
        cursor: seq_cursor_id,
        key_reg: watermark_rowid_reg,
        record_reg,
        flag: InsertFlags::new()
            .skip_last_rowid()
            .skip_all_change_counts()
            .require_seek(),
        table_name: SQLITE_SEQUENCE_TABLE_NAME.to_string(),
    });
    program.preassign_label_to_next_insn(done);
}

/// Creates the classes table, its name index, the `sqlite_sequence` table
/// and the oid watermark row in one write transaction, unless the table
/// exists. A database that already has relations without the table is
/// refused, and a read-only open can neither create nor need it: the
/// database is empty. After this returns, every relation the database will
/// ever hold has a class row from its CREATE.
pub(crate) fn ensure_classes_table(
    db: &Arc<turso_core::Database>,
    read_only: bool,
) -> crate::Result<()> {
    let conn = db.connect(turso_core::ConnectOpts::new())?;
    if table_exists(&conn, CLASSES_TABLE_NAME)? {
        return Ok(());
    }
    refuse_relations_without_catalog(&conn)?;
    if read_only {
        return Ok(());
    }
    run_internal(&conn, "BEGIN IMMEDIATE")?;
    let result = create_classes_table(&conn);
    match result {
        Ok(()) => {
            run_internal(&conn, "COMMIT")?;
            // The open loaded the schema before the table existed, so the
            // in-memory catalog is still unloaded.
            conn.reload_dialect_schema().map_err(Into::into)
        }
        Err(error) => {
            let _ = run_internal(&conn, "ROLLBACK");
            Err(error)
        }
    }
}

/// Refuses a database that has relations but no catalog to describe them.
fn refuse_relations_without_catalog(conn: &Arc<Connection>) -> crate::Result<()> {
    if conn.current_schema().user_relations().is_empty() {
        return Ok(());
    }
    Err(LimboError::ParseError(
        "the database has relations but no relation catalog, as an earlier version of tursopg \
         made it; this version does not open it"
            .to_string(),
    )
    .into())
}

fn create_classes_table(conn: &Arc<Connection>) -> crate::Result<()> {
    // Two processes can race to open the same new file; BEGIN IMMEDIATE
    // serializes them and the loser finds the table.
    if table_exists(conn, CLASSES_TABLE_NAME)? {
        return Ok(());
    }
    run_internal(conn, CREATE_CLASSES_TABLE_SQL)?;
    run_internal(conn, CREATE_CLASSES_NAME_INDEX_SQL)?;
    if !table_exists(conn, "sqlite_sequence")? {
        run_internal(conn, "CREATE TABLE sqlite_sequence(name,seq)")?;
    }
    run_internal(
        conn,
        &format!(
            "INSERT INTO sqlite_sequence (name, seq) VALUES ('{CLASSES_TABLE_NAME}', {})",
            FIRST_NORMAL_OBJECT_ID - 1
        ),
    )
}

fn table_exists(conn: &Arc<Connection>, name: &str) -> crate::Result<bool> {
    Ok(conn
        .current_schema()
        .get_btree_table(NamespaceId::ROOT, name)
        .is_some())
}

fn run_internal(conn: &Arc<Connection>, sql: &str) -> crate::Result<()> {
    conn.prepare_engine_schema_statement(sql)?
        .run_ignore_rows()
        .map_err(Into::into)
}
