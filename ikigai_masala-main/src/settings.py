"""Settings the data + domain layers need, with no dependency on an interface.

This module is where settings that belong *below* the interfaces live. The rule
is one-way: ``api/`` and the Streamlit app may import from here, and this module
imports nothing from either.

The database is AlloyDB (SQLAlchemy + pg8000). Connection details are resolved
by ``dbhandlers.DatabaseEngine`` / ``utils.secrets_manager_helper`` from, in
order: ``DATABASE_URL`` → local ``secrets.yaml`` / ``DB_*`` env (when
``EMULATE_LOCAL=True``) → GCP Secret Manager. There are therefore no DB
credentials to resolve here — that indirection lives in the secrets helper.

(Historically this module resolved Supabase URL/key + a client timeout. Supabase
has been removed; the name is kept for the two settings below and as the
downward-pointing leaf ``src/db.py`` reads.)
"""

from __future__ import annotations

import os

#: Statement timeout hint (seconds) for slow AlloyDB queries. Kept small so a
#: stuck query fails fast instead of pinning a Flask worker. Not yet wired into
#: the engine (pool_pre_ping handles dead connections); reserved for a future
#: ``connect_args`` / ``SET statement_timeout`` without another config surface.
DB_STATEMENT_TIMEOUT_SECONDS = float(os.getenv('DB_STATEMENT_TIMEOUT_SECONDS', '30'))
