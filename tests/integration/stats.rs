use turso_core::namespace::NamespaceId;
use turso_core::schema::Schema;
use turso_core::security::roles::SessionRole;
use turso_core::sync::Arc;
use turso_core::ConnectOpts;
use turso_core::Connection;
use turso_core::Database;
use turso_core::DatabaseOpts;
use turso_core::MemoryIO;
use turso_core::OpenFlags;
use turso_core::PlatformIO;
use turso_core::IO;
use turso_sqlite::SqliteDialect;

/// Regression test for #11581: once a database has `sqlite_stat1`, opening
/// a new connection must not deep-clone the whole schema just to install
/// the ANALYZE stats. Every connection keeps sharing the database schema.
#[test]
fn new_connections_share_schema_when_stat1_exists() {
    let (_dir, db) = open_db();
    create_table_and_analyze(&db);
    let mut connections = Vec::new();
    for _ in 0..5 {
        let conn = db
            .connect(ConnectOpts::new(SessionRole::Superuser))
            .unwrap();
        assert!(
            Arc::ptr_eq(&conn.current_schema(), &db.clone_schema()),
            "new connection must share the database schema instead of deep-cloning it"
        );
        connections.push(conn);
    }
    assert_eq!(connections.len(), 5);
    assert_table_t_stats(&db.clone_schema());
}

/// An existing connection that runs ANALYZE itself ends up sharing the
/// database schema and carrying the correct stats (#11581).
#[test]
fn existing_connection_shares_schema_after_analyze() {
    let (_dir, db) = open_db();
    let setup = db
        .connect(ConnectOpts::new(SessionRole::Superuser))
        .unwrap();
    setup
        .execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT, b TEXT, c INTEGER);
             CREATE INDEX t_a ON t(a, c);
             CREATE INDEX t_b ON t(b);
             INSERT INTO t (a, b, c) VALUES ('x', 'y', 1), ('z', 'w', 2);",
        )
        .unwrap();
    drop(setup);
    let conn = db
        .connect(ConnectOpts::new(SessionRole::Superuser))
        .unwrap();
    conn.execute("ANALYZE").unwrap();
    assert!(
        Arc::ptr_eq(&conn.current_schema(), &db.clone_schema()),
        "connection that ran ANALYZE must share the database schema"
    );
    assert_table_t_stats(&conn.current_schema());
    assert_table_t_stats(&db.clone_schema());
}

/// When the stats actually changed, the shared schema is updated and the
/// connection is pointed at the new shared schema (#11581).
#[test]
fn analyze_after_more_rows_updates_shared_stats() {
    let (_dir, db) = open_db();
    create_table_and_analyze(&db);
    let conn = db
        .connect(ConnectOpts::new(SessionRole::Superuser))
        .unwrap();
    conn.execute(
        "INSERT INTO t (a, b, c) VALUES ('p', 'q', 3), ('r', 's', 4), ('u', 'v', 5);
         ANALYZE",
    )
    .unwrap();
    assert!(
        Arc::ptr_eq(&conn.current_schema(), &db.clone_schema()),
        "connection must share the database schema after re-ANALYZE"
    );
    let shared = db.clone_schema();
    assert_eq!(
        shared
            .analyze_stats
            .table_stats(NamespaceId::ROOT, "t")
            .unwrap()
            .row_count,
        Some(5)
    );
    drop(shared);
    let conn2 = db
        .connect(ConnectOpts::new(SessionRole::Superuser))
        .unwrap();
    assert!(Arc::ptr_eq(&conn2.current_schema(), &db.clone_schema()));
    assert_eq!(
        conn2
            .current_schema()
            .analyze_stats
            .table_stats(NamespaceId::ROOT, "t")
            .unwrap()
            .row_count,
        Some(5)
    );
}

/// A connection holding a private copy of the schema at the same version
/// (installed here by a reparse without publishing, the shape a sync
/// checkpoint leaves behind) is not pointed at the shared schema by a
/// stats refresh: it keeps its own schema and still gets the stats
/// (#11581).
#[test]
fn private_same_version_schema_survives_analyze() {
    let (_dir, db) = open_db();
    create_table_and_analyze(&db);
    let a = db
        .connect(ConnectOpts::new(SessionRole::Superuser))
        .unwrap();
    a.force_reparse_schema_without_publish().unwrap();
    assert!(
        !Arc::ptr_eq(&a.current_schema(), &db.clone_schema()),
        "test setup must leave a private schema on the connection"
    );
    a.execute("ANALYZE").unwrap();
    assert!(
        !Arc::ptr_eq(&a.current_schema(), &db.clone_schema()),
        "refresh must not discard a private same-version schema"
    );
    assert_table_t_stats(&a.current_schema());
}

/// Control case for #11581: without `sqlite_stat1` there is nothing to
/// install, so new connections trivially keep sharing the database schema.
#[test]
fn connections_share_schema_without_stat1() {
    let (_dir, db) = open_db();
    let setup = db
        .connect(ConnectOpts::new(SessionRole::Superuser))
        .unwrap();
    setup
        .execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        .unwrap();
    drop(setup);
    let mut connections = Vec::new();
    for _ in 0..5 {
        let conn = db
            .connect(ConnectOpts::new(SessionRole::Superuser))
            .unwrap();
        assert!(
            Arc::ptr_eq(&conn.current_schema(), &db.clone_schema()),
            "new connection must share the database schema"
        );
        connections.push(conn);
    }
    assert_eq!(connections.len(), 5);
}

/// Open a fresh temporary database for the stats tests above.
fn open_db() -> (tempfile::TempDir, Arc<Database>) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stats.db");
    let io: Arc<dyn IO> = Arc::new(PlatformIO::new().unwrap());
    let db = Database::open_file_with_flags(
        io,
        path.to_str().unwrap(),
        OpenFlags::default(),
        DatabaseOpts::new(),
        None,
        Arc::new(SqliteDialect),
    )
    .unwrap();
    (dir, db)
}

/// Create table `t` (2 rows, 2 indexes) and ANALYZE it on a fresh
/// connection, which is dropped on return.
fn create_table_and_analyze(db: &Arc<Database>) {
    let setup = db
        .connect(ConnectOpts::new(SessionRole::Superuser))
        .unwrap();
    setup
        .execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT, b TEXT, c INTEGER);
             CREATE INDEX t_a ON t(a, c);
             CREATE INDEX t_b ON t(b);
             INSERT INTO t (a, b, c) VALUES ('x', 'y', 1), ('z', 'w', 2);
             ANALYZE",
        )
        .unwrap();
}

/// The stats the tests expect for table `t` after ANALYZE on its 2 rows.
fn assert_table_t_stats(schema: &Schema) {
    let stats = &schema.analyze_stats;
    assert_eq!(stats.tables.len(), 1, "only table t should have stats");
    let t = stats
        .table_stats(NamespaceId::ROOT, "t")
        .expect("table t must have stats");
    assert_eq!(t.row_count, Some(2));
    let t_a = t.index_stats.get("t_a").expect("index t_a must have stats");
    assert_eq!(t_a.total_rows, Some(2));
    let t_b = t.index_stats.get("t_b").expect("index t_b must have stats");
    assert_eq!(t_b.total_rows, Some(2));
}

fn open_memory_db() -> Arc<Database> {
    let io: Arc<dyn IO> = Arc::new(MemoryIO::new());
    Database::open_file_with_flags(
        io,
        ":memory:",
        OpenFlags::Create,
        DatabaseOpts::new(),
        None,
        Arc::new(SqliteDialect),
    )
    .unwrap()
}

fn connect(db: &Arc<Database>) -> Arc<Connection> {
    db.connect(ConnectOpts::new(SessionRole::Superuser)).unwrap()
}

fn scans(db: &Arc<Database>) -> u64 {
    db.stats_scan_count()
}

fn t_row_count(conn: &Arc<Connection>) -> Option<u64> {
    conn.current_schema()
        .analyze_stats
        .table_stats(NamespaceId::ROOT, "t")
        .and_then(|stats| stats.row_count)
}

/// A database with an analyzed table `t` of three rows, and the connection
/// that analyzed it.
fn analyzed_db() -> (Arc<Database>, Arc<Connection>) {
    let db = open_memory_db();
    let writer = connect(&db);
    writer
        .execute("CREATE TABLE t (a INTEGER PRIMARY KEY, b)")
        .unwrap();
    writer.execute("CREATE INDEX t_b ON t (b)").unwrap();
    writer
        .execute("INSERT INTO t VALUES (1, 1), (2, 2), (3, 3)")
        .unwrap();
    writer.execute("ANALYZE").unwrap();
    // The first connection after a stats-changing step may still have to
    // load; what matters is that the ones after it do not.
    let warm = connect(&db);
    assert_eq!(t_row_count(&warm), Some(3));
    (db, writer)
}

#[test]
fn stats_are_loaded_once_across_connections() {
    let (db, _writer) = analyzed_db();
    let before = scans(&db);
    for _ in 0..5 {
        let conn = connect(&db);
        assert_eq!(t_row_count(&conn), Some(3));
    }
    assert_eq!(scans(&db), before, "unchanged stats must not be rescanned");
}

#[test]
fn databases_without_stats_never_scan() {
    let db = open_memory_db();
    let conn = connect(&db);
    conn.execute("CREATE TABLE t (a INTEGER PRIMARY KEY, b)")
        .unwrap();
    conn.execute("INSERT INTO t VALUES (1, 1)").unwrap();
    for _ in 0..3 {
        let conn = connect(&db);
        assert_eq!(t_row_count(&conn), None);
    }
    assert_eq!(scans(&db), 0);
}

#[test]
fn plain_writes_keep_stats_fresh() {
    let (db, writer) = analyzed_db();
    let before = scans(&db);
    writer
        .execute("INSERT INTO t VALUES (4, 4), (5, 5)")
        .unwrap();
    let conn = connect(&db);
    // Stats describe the last ANALYZE, not the current rows.
    assert_eq!(t_row_count(&conn), Some(3));
    assert_eq!(scans(&db), before);
}

#[test]
fn analyze_refreshes_stats_for_new_connections() {
    let (db, writer) = analyzed_db();
    writer
        .execute("INSERT INTO t VALUES (4, 4), (5, 5)")
        .unwrap();
    let other = connect(&db);
    other.execute("ANALYZE").unwrap();
    let conn = connect(&db);
    assert_eq!(t_row_count(&conn), Some(5));
    let before = scans(&db);
    let conn = connect(&db);
    assert_eq!(t_row_count(&conn), Some(5));
    assert_eq!(scans(&db), before);
}

#[test]
fn direct_stat1_writes_refresh_stats_for_new_connections() {
    let (db, writer) = analyzed_db();
    writer.execute("DELETE FROM sqlite_stat1").unwrap();
    writer
        .execute("INSERT INTO sqlite_stat1 VALUES ('t', NULL, '42')")
        .unwrap();
    let conn = connect(&db);
    assert_eq!(t_row_count(&conn), Some(42));
    let before = scans(&db);
    let conn = connect(&db);
    assert_eq!(t_row_count(&conn), Some(42));
    assert_eq!(scans(&db), before);
}

#[test]
fn uncommitted_stat1_writes_are_picked_up_after_commit() {
    let (db, writer) = analyzed_db();
    writer.execute("BEGIN").unwrap();
    writer.execute("DELETE FROM sqlite_stat1").unwrap();
    writer
        .execute("INSERT INTO sqlite_stat1 VALUES ('t', NULL, '42')")
        .unwrap();
    // Connections made while the writer is open see the committed rows and
    // must not cache them as fresh.
    for _ in 0..2 {
        let conn = connect(&db);
        assert_eq!(t_row_count(&conn), Some(3));
    }
    writer.execute("COMMIT").unwrap();
    let conn = connect(&db);
    assert_eq!(t_row_count(&conn), Some(42));
}

#[test]
fn rolled_back_stat1_writes_keep_old_stats() {
    let (db, writer) = analyzed_db();
    writer.execute("BEGIN").unwrap();
    writer.execute("DELETE FROM sqlite_stat1").unwrap();
    writer.execute("ROLLBACK").unwrap();
    let conn = connect(&db);
    assert_eq!(t_row_count(&conn), Some(3));
}

#[test]
fn schema_changes_reload_stats() {
    let (db, writer) = analyzed_db();
    writer.execute("CREATE TABLE u (x)").unwrap();
    let before = scans(&db);
    let conn = connect(&db);
    assert_eq!(t_row_count(&conn), Some(3));
    assert!(scans(&db) > before, "DDL must invalidate loaded stats");
    let before = scans(&db);
    let conn = connect(&db);
    assert_eq!(t_row_count(&conn), Some(3));
    assert_eq!(scans(&db), before);
}

/// A DELETE without WHERE clears the b-tree without opening a write cursor,
/// and no INSERT follows to invalidate the stats.
#[test]
fn delete_all_stat1_rows_refreshes_stats_for_new_connections() {
    let (db, writer) = analyzed_db();
    writer.execute("DELETE FROM sqlite_stat1").unwrap();
    let conn = connect(&db);
    assert_eq!(t_row_count(&conn), None);
    let before = scans(&db);
    let conn = connect(&db);
    assert_eq!(t_row_count(&conn), None);
    assert_eq!(scans(&db), before);
}
