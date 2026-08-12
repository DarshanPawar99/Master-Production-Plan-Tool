"""Database handlers package — AlloyDB via SQLAlchemy (op-mis-backend pattern).

``DatabaseEngine`` is the process-wide SQLAlchemy engine singleton. The
Supabase-shaped query adapter lives in ``dbhandlers.pg_adapter``; the app keeps
calling ``get_supabase()`` (from ``src.db``) and gets that adapter back, so no
call site had to change when the backend moved from Supabase to AlloyDB.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from utils.secrets_manager_helper import SecretsManagerHelper


class DatabaseEngine:
    """Singleton SQLAlchemy engine for AlloyDB."""

    _instance = None
    _engine = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_engine(cls) -> Engine:
        if cls._engine is None:
            db_connection_string = SecretsManagerHelper.get_db_connection_string()
            cls._engine = create_engine(
                db_connection_string,
                pool_pre_ping=True,   # recycle stale AlloyDB connections
                future=True,
            )
        return cls._engine

    @classmethod
    def reset_for_tests(cls) -> None:
        """Drop the cached engine so the next get_engine() rebuilds it."""
        if cls._engine is not None:
            cls._engine.dispose()
        cls._engine = None
        SecretsManagerHelper.reset_for_tests()


__all__ = ["DatabaseEngine"]
