"""Integration tests for the AlloyDB query adapter against a REAL Postgres.

AlloyDB is Postgres-wire-compatible, so a local Postgres exercises the exact
SQL path production uses. These are skipped unless ``TEST_DATABASE_URL`` points
at a reachable Postgres/AlloyDB (so CI without a DB stays green)::

    TEST_DATABASE_URL=postgresql+pg8000://postgres@127.0.0.1:5433/menu_engineering \
        pytest tests/test_alloydb_adapter.py

They assert the Supabase-shaped surface the app relies on: JSONB round-trips,
DATE columns coming back as ISO strings, ``maybe_single`` semantics, RETURNING
on writes (which optimistic concurrency depends on), and the missing-relation /
undefined-column error shapes the migration-fallback paths detect.
"""

from __future__ import annotations

import datetime as dt
import os
import uuid

import pytest

_URL = os.getenv("TEST_DATABASE_URL", "").strip()
pytestmark = pytest.mark.skipif(
    not _URL, reason="set TEST_DATABASE_URL to a Postgres/AlloyDB to run"
)


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", _URL)
    import src.db as db_mod
    db_mod.reset_db_singletons_for_tests()
    from models.db import Base
    Base.metadata.create_all(db_mod.get_engine())
    sb = db_mod.get_supabase()
    yield sb
    # Clean up any rows this module created (names are uuid-prefixed).
    for tbl in ("week_signatures", "menu_history"):
        sb.table(tbl).delete().gte("client_name", "adaptertest-").lte(
            "client_name", "adaptertest-~"
        ).execute()
    sb.table("clients").delete().gte("name", "adaptertest-").lte(
        "name", "adaptertest-~"
    ).execute()
    db_mod.reset_db_singletons_for_tests()


def _name() -> str:
    return f"adaptertest-{uuid.uuid4().hex[:8]}"


def test_insert_select_eq_and_jsonb_roundtrip(client):
    name = _name()
    client.table("clients").insert({
        "name": name, "counters": [{"name": "C1", "categories": ["bread"]}],
        "city": "Pune", "serve_weekends": True, "source_pools": ["cloudera"],
    }).execute()

    row = (
        client.table("clients").select("*").eq("name", name)
        .maybe_single().execute().data
    )
    assert row is not None
    assert row["counters"] == [{"name": "C1", "categories": ["bread"]}]  # JSONB list
    assert row["source_pools"] == ["cloudera"]
    assert row["city"] == "Pune"
    assert row["serve_weekends"] is True
    assert row["version"] == 1                      # server default
    assert isinstance(row["created_at"], str)       # timestamp → ISO string


def test_maybe_single_missing_returns_none(client):
    assert (
        client.table("clients").select("*").eq("name", _name())
        .maybe_single().execute().data
    ) is None


def test_update_returns_rows_for_optimistic_concurrency(client):
    name = _name()
    client.table("clients").insert({"name": name, "counters": []}).execute()

    # Matching version → returns the updated row (non-empty .data).
    ok = (
        client.table("clients").update({"version": 2, "city": "NCR"})
        .eq("name", name).eq("version", 1).execute()
    )
    assert len(ok.data) == 1 and ok.data[0]["city"] == "NCR"

    # Stale version → matches nothing → empty .data (how the app detects a race).
    stale = (
        client.table("clients").update({"city": "X"})
        .eq("name", name).eq("version", 1).execute()
    )
    assert stale.data == []


def test_date_column_roundtrips_as_iso_string(client):
    name = _name()
    client.table("clients").insert({"name": name, "counters": []}).execute()
    client.table("menu_history").insert({
        "client_name": name, "service_date": "2026-08-03",
        "menu": {"bread": "chapati"},
    }).execute()

    rows = (
        client.table("menu_history").select("service_date, menu")
        .eq("client_name", name).in_("service_date", ["2026-08-03", "2026-08-04"])
        .execute().data
    )
    assert len(rows) == 1
    assert rows[0]["service_date"] == "2026-08-03"   # DATE → ISO string
    assert rows[0]["menu"] == {"bread": "chapati"}   # JSONB dict
    # And the value the app feeds back parses cleanly.
    assert dt.date.fromisoformat(rows[0]["service_date"]) == dt.date(2026, 8, 3)


def test_delete_returns_deleted_rows(client):
    name = _name()
    client.table("clients").insert({"name": name, "counters": []}).execute()
    out = client.table("clients").delete().eq("name", name).execute()
    assert len(out.data) == 1 and out.data[0]["name"] == name


def test_order_and_limit(client):
    # Isolate to a unique prefix so any leftover `adaptertest-` rows from other
    # tests in this module can't affect the "first by name" assertion.
    prefix = f"adaptertest-{uuid.uuid4().hex[:8]}-"
    a, b = prefix + "a", prefix + "b"
    client.table("clients").insert([
        {"name": a, "counters": []}, {"name": b, "counters": []},
    ]).execute()
    rows = (
        client.table("clients").select("name")
        .gte("name", prefix).lte("name", prefix + "~")
        .order("name").limit(1).execute().data
    )
    assert rows and rows[0]["name"] == a


def test_unknown_table_and_column_error_shapes(client):
    from dbhandlers.pg_adapter import RelationDoesNotExistError, UndefinedColumnError
    with pytest.raises(RelationDoesNotExistError) as t:
        client.table("menu_categories")
    assert t.value.code == "42P01"
    with pytest.raises(UndefinedColumnError) as c:
        client.table("clients").select("*").eq("nope", 1).execute()
    assert c.value.code == "42703"
