# Setup

Everything you need to get from a fresh clone to a running planner. The
[README quick start](../README.md#quick-start) is the 30-second version;
this file has the full story.

---

## 1. Prerequisites

- Python 3.10+
- An AlloyDB / PostgreSQL database named `menu_engineering`
- The schema applied once (see [Database schema](#3-database-schema))
- A connection source: `DATABASE_URL`, or `EMULATE_LOCAL=True` + `secrets.yaml`,
  or `GCP_PROJECT` (Secret Manager) — see [Secrets](#4-secrets)

> Migrating from Supabase? See [docs/alloydb_migration.md](alloydb_migration.md)
> for what changed (schema and features are unchanged; only the connection).

---

## 2. Install

```bash
cd ikigai_masala-main
pip install -r requirements-dev.txt   # runtime + pytest + ruff + bandit
# or `-r requirements.txt` for runtime only (prod containers)
```

---

## 3. Database schema

The whole schema is **four tables**: `clients`, `app_settings`, `menu_history`,
`week_signatures`. A client's entire cuisine config is one JSON document in
`clients.counters` (plus `city` and the other per-client columns); menu history
is one JSON row per `(client, service_date)`.

Run the setup script once against the `menu_engineering` database. It's
idempotent (`CREATE TABLE IF NOT EXISTS` / `ADD COLUMN IF NOT EXISTS`):

```bash
psql "$DATABASE_URL" -f scripts/alloydb_setup.sql
# or paste scripts/alloydb_setup.sql into AlloyDB Studio
```

`scripts/setup_all.sql` is the equivalent Supabase-editor script and additionally
migrates an older normalized database (folds the legacy `menu_categories` /
`slot_count_overrides` / `theme_overrides` tables into `clients.counters` and
reshapes the old per-dish `menu_history`).

---

## 4. Secrets

The AlloyDB connection is resolved from the first source available, in order:

1. **`DATABASE_URL`** — a full SQLAlchemy URL, used verbatim (simplest for local
   dev / scripts / CI):

   ```bash
   export DATABASE_URL="postgresql+pg8000://user:pass@host:5432/menu_engineering"
   ```

2. **Local mode** — `EMULATE_LOCAL=True` reads `secrets.yaml` (copy
   `secrets.example.yaml` → `secrets.yaml`, git-ignored), or falls back to
   `DB_USER`/`DB_PASS`/`DB_HOST`/`DB_PORT`/`DB_NAME` env vars:

   ```yaml
   # secrets.yaml
   db:
     user: postgres
     pass: postgres
     host: 127.0.0.1
     port: 5432
     name: menu_engineering
   ```

3. **Cloud** — leave `EMULATE_LOCAL` unset and set `GCP_PROJECT`; the DB creds
   come from the GCP Secret Manager secret `db-connection-config` (a JSON object
   with `user/pass/host/port/name`).

A Streamlit `.streamlit/secrets.toml` deployment still works — any of
`DATABASE_URL`, `EMULATE_LOCAL`, `GCP_PROJECT`, or the `DB_*` keys placed there
are bridged into the environment at startup.

- Never commit `secrets.yaml` or real credentials. Rotate immediately if leaked.

### Optional env vars

```toml
APP_TIMEZONE               = "Asia/Kolkata"   # default; any IANA name
LOG_FORMAT                 = "json"           # structured logs for prod
LOG_LEVEL                  = "INFO"
APP_VERSION                = "$(git rev-parse --short HEAD)"   # surfaced in /health + /
DB_STATEMENT_TIMEOUT_SECONDS = "30"           # slow-query hint; default 30s
SOLVER_GATE_ENABLED        = "false"          # solve concurrency gate; disabled by default
SOLVER_WORKERS             = "9"              # CP-SAT workers per solve when the gate is off
CORS_ALLOWED_ORIGINS       = "https://prod.example.com"   # comma-separated; defaults to loopback only
API_HOST                   = "127.0.0.1"      # loopback. Containers / prod may want 0.0.0.0
API_PORT                   = "5000"
```

`APP_TIMEZONE` decides what "today" means when the client doesn't pass an
explicit `start_date`. Change it if the kitchens you're planning for operate
in another zone — otherwise a container running in UTC will drift cooldown
windows and weekday themes by up to a day.

---

## 5. Run

```bash
streamlit run app.py
```

The Streamlit process auto-spawns the Flask API in a daemon thread on
`http://localhost:5000`. Both talk to the same AlloyDB database.

To run the API standalone (e.g. under gunicorn):

```bash
flask --app api.app run              # or python -m api.app
```
