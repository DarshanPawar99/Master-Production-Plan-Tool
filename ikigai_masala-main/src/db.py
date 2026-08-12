"""Shared database access — AlloyDB (SQLAlchemy + pg8000), one engine per process.

This module used to build a ``supabase.Client``. supabase-py is gone: the
backend is now AlloyDB. Consumers (``client_config``, ``history_manager``,
``api.app``) still import ``get_supabase`` and keep calling
``.table(...).select()...execute()`` — but what they get back is
:class:`dbhandlers.pg_adapter.AlloyDBClient`, a thin Supabase-shaped facade over
the SQLAlchemy engine. Nothing above this line had to change.

``get_supabase`` keeps its name purely for call-site/test compatibility (the
same backward-compatible-alias trick op-mis-backend used); prefer
``get_engine()`` for new code that wants raw SQLAlchemy.

Connection details are resolved by ``dbhandlers.DatabaseEngine`` /
``utils.secrets_manager_helper`` (DATABASE_URL → local YAML/env → GCP Secret
Manager). This module sits below the interfaces and imports nothing from
``api``/``ui``/``streamlit``.
"""

from __future__ import annotations

import logging
import threading

from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# Process-wide adapter singleton. Tests monkeypatch this attribute directly
# (see tests/conftest.py::fake_supabase), so keep the name stable.
_sb_client = None
_sb_lock = threading.Lock()


def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy engine (lazy singleton)."""
    from dbhandlers import DatabaseEngine
    return DatabaseEngine.get_engine()


def get_supabase():
    """Return a process-wide DB client, created lazily on first use.

    Despite the name (kept for compatibility), this is an
    :class:`~dbhandlers.pg_adapter.AlloyDBClient` talking to AlloyDB, not a
    supabase client.
    """
    global _sb_client
    if _sb_client is None:
        with _sb_lock:
            if _sb_client is None:
                from dbhandlers.pg_adapter import AlloyDBClient
                _sb_client = AlloyDBClient(get_engine())
    return _sb_client


def reset_db_singletons_for_tests() -> None:
    """Drop the cached adapter + engine so the next call rebuilds them."""
    global _sb_client
    with _sb_lock:
        _sb_client = None
    from dbhandlers import DatabaseEngine
    DatabaseEngine.reset_for_tests()
