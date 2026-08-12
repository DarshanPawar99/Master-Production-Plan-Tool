# Ikigai Masala

Constraint-based weekly menu planner for corporate meal providers. Generates
Indian menus that respect cuisine themes, item cooldowns, color variety,
per-client customizations, and history.

- **Frontend:** Streamlit
- **Backend:** Flask API (auto-started by Streamlit on port 5000)
- **Solver:** Google OR-Tools CP-SAT
- **Database:** AlloyDB / PostgreSQL (SQLAlchemy + pg8000) — clients, history, config

---

## Quick start

> First-time setup: run `scripts/alloydb_setup.sql` once against the
> `menu_engineering` database (the idempotent schema) — see
> [docs/setup.md](docs/setup.md).

```bash
cd ikigai_masala-main
pip install -r requirements-dev.txt

# one-time, against the menu_engineering database:
#   psql "$DATABASE_URL" -f scripts/alloydb_setup.sql
#   (or paste it into AlloyDB Studio)

# Local dev: point at any Postgres/AlloyDB. Two equivalent ways —
#   a) a full URL:
export DATABASE_URL="postgresql+pg8000://user:pass@host:5432/menu_engineering"
#   b) or EMULATE_LOCAL + a secrets.yaml (copy secrets.example.yaml):
#      export EMULATE_LOCAL=True
#
# In the cloud: leave EMULATE_LOCAL unset and set GCP_PROJECT — the DB creds
# come from the Secret Manager secret `db-connection-config`.

streamlit run app.py
```

---

## Documentation

- [docs/setup.md](docs/setup.md) — prerequisites, install, secrets, seed,
  every env var the app reads.
- [docs/architecture.md](docs/architecture.md) — system diagram, layer
  overview, design choices, plan / save / regenerate sequence diagrams.
- [docs/api.md](docs/api.md) — endpoint table, response shapes (plan,
  health, metrics), concurrency semantics, rules reference, data model,
  output formats.
- [docs/operations.md](docs/operations.md) — testing, CI, structured
  logs + metrics, troubleshooting table, project layout.

For a file-level symbol map optimised for Claude Code sessions, see
[`../CLAUDE.md`](../CLAUDE.md).

---

## Tests

```bash
pytest                # default (skips @slow)
pytest -m slow        # real-Excel full-pipeline tests
```

CI runs pytest + `ruff check --select=F,E9` + `bandit -ll` on every PR;
the slow suite runs on push-to-main and manual dispatch. See
[docs/operations.md](docs/operations.md#ci).
