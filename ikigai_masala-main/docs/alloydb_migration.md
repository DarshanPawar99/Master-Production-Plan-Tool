# AlloyDB migration (Supabase → AlloyDB)

The persistence backend moved from **Supabase (PostgREST)** to **AlloyDB**
(Postgres over SQLAlchemy + pg8000). The **schema, the data shapes, and every
tool feature are unchanged** — only the connection changed. supabase-py is
removed.

## What changed

| Area | Before | After |
|---|---|---|
| Driver | `supabase` (PostgREST over HTTP) | `SQLAlchemy` + `pg8000` (direct Postgres) |
| Credentials | `SUPABASE_URL` / `SUPABASE_KEY` env / `secrets.toml` | `DATABASE_URL`, or `secrets.yaml` (local), or GCP Secret Manager (cloud) |
| Schema | `scripts/setup_all.sql` (Supabase editor) | `scripts/alloydb_setup.sql` (same 4 tables, same columns) |
| Engine | `src/db.py` built a `supabase.Client` | `src/db.py` builds a SQLAlchemy engine + adapter |

The **schema is identical** to the Supabase one (`scripts/setup_all.sql`): the 4
tables `clients` (all columns — city, serve_weekends, item_cooldown_days,
source_pools, working_days, is_launch_site, shared_categories, counters),
`app_settings`, `menu_history` (one JSONB `menu` row per client-day), and
`week_signatures`.

## How the app still calls the DB (no call sites changed)

`get_supabase()` is kept as a **backward-compatible alias** (the same trick
op-mis-backend used). It now returns `dbhandlers.pg_adapter.AlloyDBClient`, a
thin facade exposing the exact `.table(...).select().eq()...execute().data`
surface the app already used, translated to SQL. So `client_config.py`,
`history_manager.py`, `application/history.py` and the health probe are
untouched. New code can use `src.db.get_engine()` for raw SQLAlchemy.

The adapter preserves the behaviours the app depends on:
- DATE / timestamp columns come back as **ISO strings** (PostgREST shape), and
  string date filters are coerced to `date` before hitting the DB.
- JSONB columns round-trip as Python dict/list.
- writes use `RETURNING`, so `update_client_atomic`'s optimistic-concurrency
  check (empty `.data` ⇒ stale) still works.
- unknown table / column raise errors carrying PG codes `42P01` / `42703` and
  "does not exist", so the migration-fallback paths (`_is_missing_relation` /
  `_is_undefined_column`) behave exactly as before.

## Configuring the connection

Resolution order (in `utils/secrets_manager_helper.py`):

1. **`DATABASE_URL`** — a full SQLAlchemy URL, used verbatim. Best for tests /
   scripts / a quick local run:
   `postgresql+pg8000://user:pass@host:5432/menu_engineering`
2. **Local mode** (`EMULATE_LOCAL=True`) — reads `secrets.yaml` if present
   (copy `secrets.example.yaml`), else `DB_USER`/`DB_PASS`/`DB_HOST`/
   `DB_PORT`/`DB_NAME` env vars.
3. **Cloud** (`EMULATE_LOCAL` unset/false) — reads the GCP Secret Manager secret
   `db-connection-config` (a JSON object with `user/pass/host/port/name`);
   needs `GCP_PROJECT`.

`secrets.yaml`:

```yaml
db:
  user: postgres
  pass: postgres
  host: 127.0.0.1
  port: 5432
  name: menu_engineering
```

## First-time schema

```bash
psql "$DATABASE_URL" -f scripts/alloydb_setup.sql
# or paste scripts/alloydb_setup.sql into AlloyDB Studio (database: menu_engineering)
```

Idempotent — safe to re-run. It creates the 4 tables and seeds `app_settings`.
(It does **not** migrate the old normalized config tables; that already happened
on Supabase, so the data you import is already in the `counters` JSONB shape.)

## Concurrency gate disabled

The solve concurrency gate (max 2 solves at once, others queue / 503 — the
"only-so-many-at-once" limiter) is **disabled by default**. Every request now
solves immediately with a fixed worker budget (`SOLVER_WORKERS`, default 9).
Re-enable it with `SOLVER_GATE_ENABLED=true` (e.g. a small single box where two
CP-SAT solves at once would exhaust RAM).

## Verifying against a real database

The adapter is covered by `tests/test_alloydb_adapter.py`, which runs against a
real Postgres/AlloyDB when `TEST_DATABASE_URL` is set (skipped otherwise so CI
without a DB stays green):

```bash
TEST_DATABASE_URL=postgresql+pg8000://postgres@127.0.0.1:5433/menu_engineering \
    pytest tests/test_alloydb_adapter.py
```

## Note on other docs

`docs/setup.md`, `docs/architecture.md`, `docs/api.md`, and `docs/operations.md`
still contain historical references to Supabase in their prose. The connection
facts above supersede them; those deep-dive docs were not rewritten line by line
in this migration.
