use crate::namespace::NamespaceId;
use crate::sync::atomic::{AtomicU64, Ordering};
use crate::sync::{Arc, Mutex, Weak};
use crate::types::IOResultOr;
use rustc_hash::FxHashMap as HashMap;

use crate::alloc::TursoVecExt;
use crate::schema::Schema;
use crate::statement::StatementOrigin;
use crate::types::IOResult;
use crate::util::normalize_ident;
use crate::{Connection, Result, Statement, TransactionState, Value};
pub const STATS_TABLE: &str = "sqlite_stat1";
const STATS_QUERY: &str = "SELECT tbl, idx, stat FROM sqlite_stat1";

/// Whether `name` is a table whose rows feed [`AnalyzeStats`] (or, for
/// `sqlite_stat4`, would once it is consumed).
pub(crate) fn is_stats_table(name: &str) -> bool {
    name.eq_ignore_ascii_case(STATS_TABLE) || name.eq_ignore_ascii_case("sqlite_stat4")
}

/// What an [`AnalyzeStats`] value was loaded against. The stats are reused by
/// a new connection only while every part of the mark still matches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatsLoadMark {
    /// [`StatsFreshness::epoch`] before the scan started.
    epoch: u64,
    /// Schema version of the snapshot the rows were interpreted against.
    schema_version: u32,
    /// WAL position `(checkpoint_seq, max_frame)` before the scan started. Only
    /// recorded for multiprocess WAL, where another process can change
    /// `sqlite_stat1` without this process seeing it in `epoch`.
    wal_pos: Option<(u32, u64)>,
}

/// Database-wide bookkeeping that decides whether the stats carried by the
/// shared schema are still what `sqlite_stat1` contains.
///
/// A scan is skipped on connect only if all of these hold, see
/// [`stats_are_fresh`]:
/// - the stats were loaded at the schema version the connection starts with
///   (DDL and ANALYZE that creates `sqlite_stat1` bump it);
/// - no transaction in this process wrote `sqlite_stat1`/`sqlite_stat4` since
///   (`epoch`, bumped when such a transaction ends) and none is in flight;
/// - for multiprocess WAL, nobody committed anything since: the pager exposes
///   no per-table change marker, so any commit by another process counts.
pub(crate) struct StatsFreshness {
    epoch: AtomicU64,
    /// Connections that wrote a stats table and have not ended that
    /// transaction yet.
    writers: Mutex<Vec<Weak<Connection>>>,
    /// Number of `sqlite_stat1` scans started, for tests.
    scans: AtomicU64,
}

impl StatsFreshness {
    pub(crate) fn new() -> Self {
        Self {
            epoch: AtomicU64::new(0),
            writers: Mutex::new(Vec::new()),
            scans: AtomicU64::new(0),
        }
    }

    pub(crate) fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::SeqCst)
    }

    /// Stats loaded earlier may no longer match `sqlite_stat1`.
    pub(crate) fn invalidate(&self) {
        self.epoch.fetch_add(1, Ordering::SeqCst);
    }

    pub(crate) fn scan_count(&self) -> u64 {
        self.scans.load(Ordering::SeqCst)
    }

    /// Whether a transaction that wrote a stats table is still open. A writer
    /// whose transaction ended without passing through
    /// [`finish_stats_table_write`] (error paths, dropped connection) is
    /// treated as finished and invalidates the stats.
    fn writer_in_flight(&self) -> bool {
        let mut writers = self.writers.lock();
        writers.retain(|writer| {
            let Some(conn) = writer.upgrade() else {
                self.invalidate();
                return false;
            };
            let mut written = conn.stats_tables_written.lock();
            // The connection wrote this database's stats tables only while
            // its entry is in the list; finishing the write removes it.
            let Some(pos) = written
                .iter()
                .position(|db| std::ptr::eq(&db.stats_freshness, self))
            else {
                return false;
            };
            if conn.get_tx_state() == TransactionState::None {
                written.remove(pos);
                self.invalidate();
                false
            } else {
                true
            }
        });
        !writers.is_empty()
    }
}

/// A statement of `conn` is about to write `sqlite_stat1`/`sqlite_stat4` of
/// database `db_id` (main or attached). The stats cached by that database's own
/// connections are what goes stale, so it is tracked per [`Database`].
pub(crate) fn note_stats_table_write(conn: &Arc<Connection>, db_id: usize) {
    let db = conn.get_source_database(db_id);
    {
        let mut written = conn.stats_tables_written.lock();
        if written.iter().any(|known| Arc::ptr_eq(known, &db)) {
            return;
        }
        written.push(db.clone());
    }
    // Not under the connection's lock: `StatsFreshness::writer_in_flight` takes
    // the two in the opposite order.
    db.stats_freshness.writers.lock().push(Arc::downgrade(conn));
}

/// The transaction of `conn` ended, committed or rolled back. Called once the
/// commit is visible to other connections.
pub(crate) fn finish_stats_table_write(conn: &Connection) {
    let written = std::mem::take(&mut *conn.stats_tables_written.lock());
    for db in written {
        db.stats_freshness.invalidate();
    }
}

/// The WAL position other processes' commits would move, or `None` when the
/// database cannot be shared with another process.
fn observed_wal_pos(conn: &Connection) -> Result<Option<(u32, u64)>> {
    if !conn.db.experimental_multiprocess_wal_enabled() {
        return Ok(None);
    }
    let pager = conn.pager.load();
    if pager.holds_read_lock() {
        return Ok(Some(pager.wal_pos()));
    }
    pager.begin_read_tx()?;
    let pos = pager.wal_pos();
    pager.end_read_tx();
    Ok(Some(pos))
}

/// Whether the stats the connection inherited from the shared schema are still
/// what `sqlite_stat1` contains, so the connect-time scan can be skipped.
fn stats_are_fresh(conn: &Connection) -> bool {
    let (mark, schema_version) = {
        let schema = conn.schema.read();
        (schema.analyze_stats.loaded, schema.schema_version)
    };
    let Some(mark) = mark else {
        return false;
    };
    let freshness = &conn.db.stats_freshness;
    if freshness.writer_in_flight() {
        return false;
    }
    mark.schema_version == schema_version
        && mark.epoch == freshness.epoch()
        && matches!(observed_wal_pos(conn), Ok(pos) if pos == mark.wal_pos)
}

/// Statistics produced by ANALYZE for a single index.
#[derive(Clone, Debug, Default)]
pub struct IndexStat {
    /// Estimated total number of rows in the table/index when the stat was collected.
    pub total_rows: Option<u64>,
    /// Average number of rows per distinct key prefix, for each leftmost prefix
    /// of the index columns.
    ///
    /// These values come directly from sqlite_stat1's stat column (after the
    /// first number which is total_rows). For a stat string "1000 100 10 1":
    /// - total_rows = 1000
    /// - avg_rows_per_distinct_prefix = [100, 10, 1]
    ///
    /// Entry at position `i` means: on average, this many rows share the same
    /// values in the first `i + 1` columns of the index. Lower values indicate
    /// higher selectivity (more distinct prefixes).
    ///
    /// To compute number of distinct values (NDV) for prefix i:
    ///   ndv = total_rows / avg_rows_per_distinct_prefix[i]
    pub avg_rows_per_distinct_prefix: Vec<u64>,
}

/// Statistics produced by ANALYZE for a single BTree table.
#[derive(Clone, Debug, Default)]
pub struct TableStat {
    /// Estimated row count for the table (sqlite_stat1 entry with a NULL index name).
    pub row_count: Option<u64>,
    /// Per-index statistics keyed by normalized index name.
    pub index_stats: HashMap<String, IndexStat>,
}

impl TableStat {
    /// Get or create the per-index statistics bucket for the given index name.
    pub fn index_stats_mut(&mut self, index_name: &str) -> &mut IndexStat {
        let index_name = normalize_ident(index_name);
        self.index_stats.entry(index_name).or_default()
    }
}

/// Container for ANALYZE statistics across the schema.
#[derive(Clone, Debug, Default)]
pub struct AnalyzeStats {
    /// Per-table statistics keyed by namespace and normalized table name.
    pub tables: HashMap<(NamespaceId, String), TableStat>,
    /// Set when these stats are a complete load of `sqlite_stat1`, which is
    /// what lets a new connection reuse them.
    loaded: Option<StatsLoadMark>,
}

impl AnalyzeStats {
    pub fn needs_refresh(&self) -> bool {
        self.tables.is_empty()
    }
    /// Get the statistics for a table, if present.
    pub fn table_stats(&self, namespace: NamespaceId, table_name: &str) -> Option<&TableStat> {
        let table_name = normalize_ident(table_name);
        self.tables.get(&(namespace, table_name))
    }

    /// Get or create the statistics bucket for a table.
    pub fn table_stats_mut(&mut self, namespace: NamespaceId, table_name: &str) -> &mut TableStat {
        let table_name = normalize_ident(table_name);
        self.tables.entry((namespace, table_name)).or_default()
    }

    /// Remove all statistics for a table.
    pub fn remove_table(&mut self, namespace: NamespaceId, table_name: &str) {
        let table_name = normalize_ident(table_name);
        self.tables.remove(&(namespace, table_name));
    }

    /// Remove statistics for a specific index on a table.
    pub fn remove_index(&mut self, namespace: NamespaceId, table_name: &str, index_name: &str) {
        let table_name = normalize_ident(table_name);
        let index_name = normalize_ident(index_name);
        if let Some(table_stats) = self.tables.get_mut(&(namespace, table_name)) {
            table_stats.index_stats.remove(&index_name);
        }
    }
}

/// Best-effort refresh of the connection's in-memory ANALYZE stats.
///
/// Blocking form of [`refresh_analyze_stats_nonblock`]: the same state machine,
/// driven to completion here by pumping IO. That is only correct when whatever
/// the `sqlite_stat1` scan waits for can make progress without the caller's
/// help (real I/O, or another thread). A host that schedules connections
/// cooperatively on one thread must drive the non-blocking variant itself,
/// because an explicit yield is a request to run *other* connections, which
/// `io.step()` never does; see `Database::connect_async`.
pub fn refresh_analyze_stats(conn: &Arc<Connection>) {
    let io = conn.db.io.clone();
    let mut state = RefreshAnalyzeStatsState::Connect;
    loop {
        let pending = match refresh_analyze_stats_nonblock(conn, &mut state) {
            Ok(IOResult::Done(())) => return,
            Ok(IOResult::IO(pending)) => pending,
            Err(e) => {
                tracing::warn!("Failed to refresh analyze stats: {e}");
                return;
            }
        };
        // An explicit yield carries nothing to wait for; give the IO one turn,
        // as the blocking runner always did, and step the scan again.
        let advanced = if pending.is_explicit_yield() {
            io.step()
        } else {
            pending.wait(io.as_ref())
        };
        if let Err(e) = advanced {
            tracing::warn!("Failed to refresh analyze stats: {e}");
            return;
        }
    }
}

/// Carries the in-progress `sqlite_stat1` scan across IO yields for
/// [`refresh_analyze_stats_nonblock`].
#[derive(Default)]
pub enum RefreshAnalyzeStatsState {
    /// Like `Start`, but first reuses the stats the connection inherited from
    /// the shared schema if they are still fresh. Only for a new connection.
    Connect,
    #[default]
    Start,
    Running {
        stmt: Box<Statement>,
        schema_snapshot: Arc<Schema>,
        stats: AnalyzeStats,
        mark: StatsLoadMark,
    },
}

/// Non-blocking variant of [`refresh_analyze_stats`]: best-effort refresh of
/// the connection's in-memory ANALYZE stats from `sqlite_stat1`, yielding IO
/// via the supplied state instead of pumping it. Errors are swallowed (matches
/// the blocking variant's best-effort contract).
pub fn refresh_analyze_stats_nonblock(
    conn: &Arc<Connection>,
    st: &mut RefreshAnalyzeStatsState,
) -> IOResultOr<()> {
    loop {
        match st {
            RefreshAnalyzeStatsState::Connect | RefreshAnalyzeStatsState::Start => {
                let reuse_fresh = matches!(st, RefreshAnalyzeStatsState::Connect);
                if !conn.is_db_initialized() || conn.is_nested_stmt() {
                    return Ok(IOResult::Done(()));
                }
                if matches!(conn.get_tx_state(), TransactionState::Write { .. }) {
                    return Ok(IOResult::Done(()));
                }
                // Sampled before the scan: anything that commits after this
                // point makes the mark stale instead of being missed.
                let epoch = conn.db.stats_freshness.epoch();
                let schema_snapshot = { conn.schema.read().clone() };
                if schema_snapshot
                    .get_btree_table(NamespaceId::ROOT, STATS_TABLE)
                    .is_none()
                {
                    return Ok(IOResult::Done(()));
                }
                if reuse_fresh && stats_are_fresh(conn) {
                    *st = RefreshAnalyzeStatsState::Start;
                    return Ok(IOResult::Done(()));
                }
                let mark = StatsLoadMark {
                    epoch,
                    schema_version: schema_snapshot.schema_version,
                    wal_pos: observed_wal_pos(conn).unwrap_or(None),
                };
                conn.db.stats_freshness.scans.fetch_add(1, Ordering::SeqCst);
                let mv_tx = conn.get_mv_tx();
                let mut stmt =
                    conn.prepare_with_origin(STATS_QUERY, StatementOrigin::EngineMetadataRead)?;
                stmt.set_mv_tx(mv_tx);
                *st = RefreshAnalyzeStatsState::Running {
                    stmt: Box::new(stmt),
                    schema_snapshot,
                    stats: AnalyzeStats::default(),
                    mark,
                };
            }
            RefreshAnalyzeStatsState::Running {
                stmt,
                schema_snapshot,
                stats,
                mark,
            } => {
                let scan = load_sqlite_stat1_rows_nonblock(stmt, schema_snapshot, stats);
                match scan {
                    Ok(IOResult::IO(io)) => return Ok(IOResult::IO(io)),
                    Ok(IOResult::Done(())) => {
                        let mut stats = std::mem::take(stats);
                        stats.loaded = Some(*mark);
                        if let Err(e) = install_analyze_stats(conn, stats) {
                            tracing::warn!("Failed to refresh analyze stats: {e}");
                        }
                        *st = RefreshAnalyzeStatsState::Start;
                        return Ok(IOResult::Done(()));
                    }
                    Err(e) => {
                        tracing::warn!("Failed to refresh analyze stats: {e}");
                        *st = RefreshAnalyzeStatsState::Start;
                        return Ok(IOResult::Done(()));
                    }
                }
            }
        }
    }
}

/// Store freshly gathered stats in the connection's schema and, when the shared
/// database schema is the same schema version, in the shared schema too.
///
/// In MVCC mode the commit that wrote `sqlite_stat1` publishes the connection's
/// schema to the shared schema before the stats are gathered. If only the
/// connection copy were updated, the two copies would differ while having the
/// same version, and the next statement would adopt the shared copy and lose
/// the new stats.
///
/// The connection schema lock is released before the shared schema lock is
/// taken: other code paths lock them in the opposite order.
fn install_analyze_stats(conn: &Arc<Connection>, stats: AnalyzeStats) -> Result<()> {
    let schema_version = conn.with_schema_mut(|schema| {
        schema.analyze_stats = stats.clone();
        schema.schema_version
    })?;
    let mut shared = conn.db.schema.lock();
    if shared.schema_version == schema_version {
        Schema::try_make_mut(&mut shared)?.analyze_stats = stats;
    }
    Ok(())
}

/// Non-blocking row scan shared by [`refresh_analyze_stats_nonblock`]. Steps the
/// prepared `sqlite_stat1` statement, accumulating into `stats`.
fn load_sqlite_stat1_rows_nonblock(
    stmt: &mut Statement,
    schema: &Schema,
    stats: &mut AnalyzeStats,
) -> crate::types::IOResultOr<()> {
    crate::return_if_io!(
        stmt.run_with_row_callback_nonblock(|row| { load_sqlite_stat1_row(row, schema, stats) })
    );
    Ok(crate::types::IOResult::Done(()))
}

/// Apply a single `sqlite_stat1` row to the accumulating [`AnalyzeStats`].
/// Shared by the blocking and non-blocking scanners.
fn load_sqlite_stat1_row(
    row: &crate::vdbe::Row,
    schema: &Schema,
    stats: &mut AnalyzeStats,
) -> Result<()> {
    let (namespace, table_name) = schema.demangle_name(row.get::<&str>(0)?);
    let idx_value = row.get::<&Value>(1)?;
    let stat_value = row.get::<&Value>(2)?;

    let idx_name = match idx_value {
        Value::Null => None,
        Value::Text(s) => Some(s.as_str()),
        _ => None,
    };
    let stat = match stat_value {
        Value::Text(s) => s.as_str(),
        _ => return Ok(()),
    };

    // Skip if table is not a regular B-tree.
    if schema.get_btree_table(namespace, table_name).is_none() {
        return Ok(());
    }
    let Some(numbers) = parse_stat_numbers(stat) else {
        return Ok(());
    };
    if numbers.is_empty() {
        return Ok(());
    }
    if idx_name.is_none() {
        if let Some(total_rows) = numbers.first().copied() {
            stats.table_stats_mut(namespace, table_name).row_count = Some(total_rows);
        }
        return Ok(());
    }

    // Index-level entry: only keep if the index exists on this table.
    let idx_name = normalize_ident(idx_name.unwrap());
    if schema.get_index(namespace, table_name, &idx_name).is_none() {
        return Ok(());
    }

    let total_rows = numbers.first().copied();
    {
        let idx_stats = stats
            .table_stats_mut(namespace, table_name)
            .index_stats_mut(&idx_name);
        idx_stats.total_rows = total_rows;
        idx_stats.avg_rows_per_distinct_prefix = numbers.iter().skip(1).copied().collect();
    }

    // If we didn't see a table-level row yet, seed row_count from index stats.
    if let Some(total_rows) = total_rows {
        let table_stats = stats.table_stats_mut(namespace, table_name);
        if table_stats.row_count.is_none() {
            table_stats.row_count = Some(total_rows);
        }
    }
    Ok(())
}

fn parse_stat_numbers(stat: &str) -> Option<Vec<u64>> {
    stat.split_whitespace()
        .map(|s| s.parse::<u64>().ok())
        .collect()
}

/// Statistics accumulator for ANALYZE.
#[derive(Debug, Clone)]
pub struct StatAccum {
    /// Number of columns in the index (not including rowid)
    pub n_col: usize,
    /// Total number of rows seen
    pub n_row: u64,
    /// Distinct counts for each column prefix.
    /// distinct[i] = number of distinct values for columns 0..=i
    pub distinct: Vec<u64>,
}

impl StatAccum {
    pub fn new(n_col: usize) -> Self {
        Self {
            n_col,
            n_row: 0,
            distinct: vec![0; n_col],
        }
    }

    /// Push a row, indicating which column (0-indexed) is the first to differ
    /// from the previous row. If this is the first row, pass 0.
    ///
    /// i_chng is the index of the leftmost column that changed:
    /// - 0 means column 0 changed (or first row)
    /// - 1 means columns 0 was same, column 1 changed
    /// - n_col means all columns were the same (duplicate row)
    pub fn push(&mut self, i_chng: usize) {
        self.n_row += 1;
        // Increment distinct counts for columns i_chng and onwards
        // because if column i changed, then prefixes (0..=i), (0..=i+1), etc. all have a new distinct value
        for i in i_chng..self.n_col {
            self.distinct[i] += 1;
        }
    }

    /// Get the stat1 string: "total avg1 avg2 ..."
    /// where avgN = ceil(total / distinctN)
    pub fn get_stat1(&self) -> String {
        if self.n_row == 0 {
            return String::new();
        }

        let mut parts = vec![self.n_row.to_string()];
        for &d in &self.distinct {
            let avg = if d > 0 {
                self.n_row.div_ceil(d)
            } else {
                self.n_row
            };
            parts.push(avg.to_string());
        }
        parts.join(" ")
    }

    /// Serialize to bytes for storage in a blob register.
    pub fn to_bytes(&self) -> crate::ValueBlob {
        let mut bytes =
            <crate::ValueBlob as TursoVecExt<u8>>::with_capacity(8 + 8 + 8 * self.n_col);
        bytes.extend_from_slice(&(self.n_col as u64).to_le_bytes());
        bytes.extend_from_slice(&self.n_row.to_le_bytes());
        for &d in &self.distinct {
            bytes.extend_from_slice(&d.to_le_bytes());
        }
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 16 {
            return None;
        }
        let n_col = u64::from_le_bytes(bytes[0..8].try_into().ok()?) as usize;
        let n_row = u64::from_le_bytes(bytes[8..16].try_into().ok()?);

        if bytes.len() < 16 + 8 * n_col {
            return None;
        }
        let mut distinct = Vec::with_capacity(n_col);
        for i in 0..n_col {
            let start = 16 + i * 8;
            let d = u64::from_le_bytes(bytes[start..start + 8].try_into().ok()?);
            distinct.push(d);
        }
        Some(Self {
            n_col,
            n_row,
            distinct,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{is_stats_table, parse_stat_numbers};
    use crate::sync::Arc;
    use crate::{
        ConnectOpts, Connection, Database, DatabaseOpts, EngineDialect, MemoryIO, NamespaceId,
        OpenFlags, IO,
    };

    #[test]
    fn parse_stat_numbers_basic() {
        assert_eq!(parse_stat_numbers("10 5 3 1").unwrap(), vec![10, 5, 3, 1]);
        assert_eq!(parse_stat_numbers("  42\t7 ").unwrap(), vec![42, 7]);
        assert!(parse_stat_numbers("abc 1").is_none());
    }

    #[test]
    fn stats_table_names() {
        assert!(is_stats_table("sqlite_stat1"));
        assert!(is_stats_table("SQLITE_STAT4"));
        assert!(!is_stats_table("sqlite_stat2"));
        assert!(!is_stats_table("t"));
    }

    fn open_db() -> Arc<Database> {
        let io: Arc<dyn IO> = Arc::new(MemoryIO::new());
        Database::open_file_with_flags(
            io,
            ":memory:",
            OpenFlags::Create,
            DatabaseOpts::new(),
            None,
            Arc::new(EngineDialect),
        )
        .unwrap()
    }

    fn connect(db: &Arc<Database>) -> Arc<Connection> {
        db.connect(ConnectOpts::new()).unwrap()
    }

    fn scans(db: &Arc<Database>) -> u64 {
        db.stats_freshness.scan_count()
    }

    fn t_row_count(conn: &Arc<Connection>) -> Option<u64> {
        conn.schema
            .read()
            .analyze_stats
            .table_stats(NamespaceId::ROOT, "t")
            .and_then(|stats| stats.row_count)
    }

    /// A database with an analyzed table `t` of three rows, and the connection
    /// that analyzed it.
    fn analyzed_db() -> (Arc<Database>, Arc<Connection>) {
        let db = open_db();
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
        let db = open_db();
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

    /// An analyzed file database `aux.db` (kept open by the returned handles),
    /// and a separate in-memory connection with it attached as `aux`.
    fn attached_analyzed_db(
        dir: &std::path::Path,
    ) -> (Arc<Database>, Arc<Connection>, Arc<Connection>) {
        let aux_path = dir.join("aux.db");
        let aux_path = aux_path.to_str().unwrap();
        let io: Arc<dyn IO> = Arc::new(crate::io::PlatformIO::new().unwrap());
        let aux_db = Database::open_file_with_flags(
            io,
            aux_path,
            OpenFlags::Create,
            DatabaseOpts::new(),
            None,
            Arc::new(EngineDialect),
        )
        .unwrap();
        let direct = connect(&aux_db);
        direct
            .execute("CREATE TABLE t (a INTEGER PRIMARY KEY, b)")
            .unwrap();
        direct.execute("CREATE INDEX t_b ON t (b)").unwrap();
        direct
            .execute("INSERT INTO t VALUES (1, 1), (2, 2), (3, 3)")
            .unwrap();
        direct.execute("ANALYZE").unwrap();
        assert_eq!(t_row_count(&connect(&aux_db)), Some(3));

        let main_io: Arc<dyn IO> = Arc::new(MemoryIO::new());
        let main_db = Database::open_file_with_flags(
            main_io,
            ":memory:",
            OpenFlags::Create,
            DatabaseOpts::new().with_attach(true),
            None,
            Arc::new(EngineDialect),
        )
        .unwrap();
        let main = connect(&main_db);
        main.execute(format!("ATTACH '{aux_path}' AS aux")).unwrap();
        (aux_db, direct, main)
    }

    #[test]
    fn attached_stat1_writes_refresh_stats_for_the_attached_database() {
        let dir = tempfile::tempdir().unwrap();
        let (aux_db, _direct, main) = attached_analyzed_db(dir.path());
        let before = scans(&aux_db);
        main.execute("DELETE FROM aux.sqlite_stat1 WHERE tbl = 't'")
            .unwrap();
        let conn = connect(&aux_db);
        assert_eq!(t_row_count(&conn), None);
        assert!(scans(&aux_db) > before, "the write must invalidate aux.db");
    }

    #[test]
    fn attached_delete_all_stat1_rows_refreshes_stats_for_the_attached_database() {
        let dir = tempfile::tempdir().unwrap();
        let (aux_db, _direct, main) = attached_analyzed_db(dir.path());
        main.execute("DELETE FROM aux.sqlite_stat1").unwrap();
        let conn = connect(&aux_db);
        assert_eq!(t_row_count(&conn), None);
    }

    #[test]
    fn attached_uncommitted_stat1_writes_are_not_cached_as_fresh() {
        let dir = tempfile::tempdir().unwrap();
        let (aux_db, _direct, main) = attached_analyzed_db(dir.path());
        main.execute("BEGIN").unwrap();
        main.execute("DELETE FROM aux.sqlite_stat1").unwrap();
        for _ in 0..2 {
            assert_eq!(t_row_count(&connect(&aux_db)), Some(3));
        }
        main.execute("COMMIT").unwrap();
        assert_eq!(t_row_count(&connect(&aux_db)), None);
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
}
