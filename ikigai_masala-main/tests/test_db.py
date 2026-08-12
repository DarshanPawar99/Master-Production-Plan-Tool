"""Tests for ``src.db`` — the AlloyDB engine + Supabase-shaped adapter wiring.

supabase-py is gone: ``get_supabase()`` now returns an
``dbhandlers.pg_adapter.AlloyDBClient`` over a lazily-built SQLAlchemy engine.
These tests pin the lazy-singleton behaviour and that the connection string is
resolved from the secrets helper (via a ``DATABASE_URL`` override here so no
real server is touched).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_db_singletons():
    import src.db as db_mod
    db_mod.reset_db_singletons_for_tests()
    yield
    db_mod.reset_db_singletons_for_tests()


class TestGetSupabaseReturnsAdapter:
    def test_returns_alloydb_client(self, monkeypatch):
        # A syntactically valid URL so create_engine succeeds without connecting
        # (engines are lazy — no socket until a statement runs).
        monkeypatch.setenv(
            "DATABASE_URL", "postgresql+pg8000://u:p@127.0.0.1:5432/menu_engineering"
        )
        from dbhandlers.pg_adapter import AlloyDBClient
        import src.db as db_mod

        client = db_mod.get_supabase()
        assert isinstance(client, AlloyDBClient)

    def test_lazy_singleton_is_reused(self, monkeypatch):
        monkeypatch.setenv(
            "DATABASE_URL", "postgresql+pg8000://u:p@127.0.0.1:5432/menu_engineering"
        )
        import src.db as db_mod
        first = db_mod.get_supabase()
        second = db_mod.get_supabase()
        assert first is second

    def test_get_engine_is_process_singleton(self, monkeypatch):
        monkeypatch.setenv(
            "DATABASE_URL", "postgresql+pg8000://u:p@127.0.0.1:5432/menu_engineering"
        )
        import src.db as db_mod
        assert db_mod.get_engine() is db_mod.get_engine()


class TestUnknownTableRaisesMissingRelation:
    """The adapter must raise a 42P01-coded error for an unknown table so the
    app's migration-fallback path (``_is_missing_relation``) still fires."""

    def test_unknown_table_error_shape(self, monkeypatch):
        monkeypatch.setenv(
            "DATABASE_URL", "postgresql+pg8000://u:p@127.0.0.1:5432/menu_engineering"
        )
        import src.db as db_mod
        from dbhandlers.pg_adapter import RelationDoesNotExistError
        from src.client.client_config import _is_missing_relation

        client = db_mod.get_supabase()
        with pytest.raises(RelationDoesNotExistError) as exc:
            client.table("menu_categories")  # a legacy table, not in the schema
        assert exc.value.code == "42P01"
        assert "does not exist" in str(exc.value)
        assert _is_missing_relation(exc.value)
