"""A Supabase-shaped query adapter backed by SQLAlchemy / AlloyDB.

The tool was written against supabase-py's fluent builder::

    sb.table('clients').select('*').eq('name', n).maybe_single().execute().data

When the backend moved from Supabase (PostgREST) to AlloyDB (Postgres over
SQLAlchemy + pg8000), rewriting every one of those call sites would have meant
touching ~6 modules and re-writing the 48 test files that lean on the same
interface via ``tests/fake_supabase.py``. Instead, ``get_supabase()`` now hands
back :class:`AlloyDBClient`, which implements the exact subset of that builder
the app uses and translates it to SQL. supabase-py itself is gone.

Supported surface (only what the app exercises):

* ``.table(name)`` → select / insert / update / delete
* filters ``.eq() .gte() .lte() .in_()``  (AND-combined, like PostgREST)
* ``.order(col, desc=?) .limit(n) .maybe_single()``
* ``.execute()`` → object with ``.data`` (list for selects, dict/None for
  ``maybe_single``, list of affected rows for insert/update/delete)

Fidelity notes that matter for the app's behaviour:

* DATE columns are returned as ISO strings and TIMESTAMPs as ISO strings —
  the same shape PostgREST produced, which the history layer relies on
  (``dt.date.fromisoformat(row['service_date'])``). String date filter values
  are coerced back to ``datetime.date`` before hitting the DB.
* An unknown table or column raises an error whose message contains
  "does not exist" (and carries a PG ``code``), so the app's
  ``_is_missing_relation`` / ``_is_undefined_column`` migration-fallback paths
  behave exactly as they did against Supabase.
* JSONB columns round-trip as Python dict/list via SQLAlchemy's JSONB type.
"""

from __future__ import annotations

import datetime as dt
import decimal
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import Date, and_, select
from sqlalchemy.engine import Engine
from sqlalchemy.schema import MetaData, Table


# ---------------------------------------------------------------------------
# Errors shaped like the psycopg / PostgREST ones the app already detects
# ---------------------------------------------------------------------------

class RelationDoesNotExistError(Exception):
    """Raised for ``.table(<unknown>)`` — mirrors PG 42P01."""

    code = "42P01"


class UndefinedColumnError(Exception):
    """Raised for a filter / payload naming an unknown column — mirrors PG 42703."""

    code = "42703"


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

class _Response:
    __slots__ = ("data",)

    def __init__(self, data):
        self.data = data


def _jsonify_scalar(value: Any) -> Any:
    """Match PostgREST's JSON projection for the types we read back."""
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        # Whole numbers → int (matches how the app treats version, etc.)
        return int(value) if value == value.to_integral_value() else float(value)
    return value


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

class _Query:
    def __init__(self, engine: Engine, table: Table):
        self._engine = engine
        self._table = table
        self._filters: List[tuple] = []
        self._order: Optional[str] = None
        self._order_desc = False
        self._limit: Optional[int] = None
        self._single = False
        self._mode = "select"
        self._payload: Any = None

    # -- column / value helpers -------------------------------------------

    def _column(self, name: str):
        col = self._table.c.get(name)
        if col is None:
            raise UndefinedColumnError(
                f'column "{name}" of relation "{self._table.name}" does not exist'
            )
        return col

    def _coerce(self, name: str, value: Any) -> Any:
        col = self._column(name)
        if isinstance(col.type, Date) and isinstance(value, str):
            # PostgREST accepted 'YYYY-MM-DD' strings; SQLAlchemy's Date wants
            # a date object with pg8000.
            return dt.date.fromisoformat(value)
        return value

    # -- filters -----------------------------------------------------------

    def select(self, *_args, **_kwargs):
        self._mode = "select"
        return self

    def eq(self, col: str, val: Any):
        self._filters.append(("eq", col, val))
        return self

    def gte(self, col: str, val: Any):
        self._filters.append(("gte", col, val))
        return self

    def lte(self, col: str, val: Any):
        self._filters.append(("lte", col, val))
        return self

    def in_(self, col: str, values: Iterable[Any]):
        self._filters.append(("in", col, list(values)))
        return self

    def order(self, col: str, desc: bool = False, **_kwargs):
        self._order = col
        self._order_desc = bool(desc)
        return self

    def limit(self, n: int):
        self._limit = int(n)
        return self

    def maybe_single(self):
        self._single = True
        return self

    # -- mutations ---------------------------------------------------------

    def insert(self, payload):
        self._mode = "insert"
        self._payload = payload
        return self

    def update(self, payload: Dict[str, Any]):
        self._mode = "update"
        self._payload = payload
        return self

    def delete(self):
        self._mode = "delete"
        return self

    # -- SQL assembly ------------------------------------------------------

    def _where(self):
        clauses = []
        for op, col_name, val in self._filters:
            col = self._column(col_name)
            if op == "eq":
                clauses.append(col == self._coerce(col_name, val))
            elif op == "gte":
                clauses.append(col >= self._coerce(col_name, val))
            elif op == "lte":
                clauses.append(col <= self._coerce(col_name, val))
            elif op == "in":
                clauses.append(col.in_([self._coerce(col_name, v) for v in val]))
            else:  # pragma: no cover — defensive
                raise RuntimeError(f"Unsupported filter op: {op}")
        return clauses

    def _coerce_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {k: self._coerce(k, v) for k, v in row.items()}

    def _rows_from(self, result) -> List[Dict[str, Any]]:
        return [
            {k: _jsonify_scalar(v) for k, v in m.items()}
            for m in result.mappings()
        ]

    # -- terminal ----------------------------------------------------------

    def execute(self) -> _Response:
        if self._mode == "select":
            stmt = select(self._table)
            clauses = self._where()
            if clauses:
                stmt = stmt.where(and_(*clauses))
            if self._order:
                oc = self._column(self._order)
                stmt = stmt.order_by(oc.desc() if self._order_desc else oc.asc())
            if self._limit is not None:
                stmt = stmt.limit(self._limit)
            with self._engine.connect() as conn:
                rows = self._rows_from(conn.execute(stmt))
            if self._single:
                return _Response(rows[0] if rows else None)
            return _Response(rows)

        if self._mode == "insert":
            payload = self._payload
            records = payload if isinstance(payload, list) else [payload]
            if not records:
                return _Response([])
            values = [self._coerce_row(r) for r in records]
            stmt = self._table.insert().values(values).returning(*self._table.c)
            with self._engine.begin() as conn:
                rows = self._rows_from(conn.execute(stmt))
            return _Response(rows)

        if self._mode == "update":
            stmt = self._table.update()
            clauses = self._where()
            if clauses:
                stmt = stmt.where(and_(*clauses))
            stmt = stmt.values(self._coerce_row(self._payload)).returning(*self._table.c)
            with self._engine.begin() as conn:
                rows = self._rows_from(conn.execute(stmt))
            return _Response(rows)

        if self._mode == "delete":
            stmt = self._table.delete()
            clauses = self._where()
            if clauses:
                stmt = stmt.where(and_(*clauses))
            stmt = stmt.returning(*self._table.c)
            with self._engine.begin() as conn:
                rows = self._rows_from(conn.execute(stmt))
            return _Response(rows)

        raise RuntimeError(f"Unknown query mode: {self._mode}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Table + client
# ---------------------------------------------------------------------------

class _Table:
    def __init__(self, engine: Engine, table: Table):
        self._engine = engine
        self._table = table

    def _q(self) -> _Query:
        return _Query(self._engine, self._table)

    def select(self, *args, **kwargs):
        return self._q().select(*args, **kwargs)

    def insert(self, payload):
        return self._q().insert(payload)

    def update(self, payload):
        return self._q().update(payload)

    def delete(self):
        return self._q().delete()


class AlloyDBClient:
    """Drop-in stand-in for ``supabase.Client`` backed by AlloyDB.

    Table definitions come from ``models.db.Base.metadata`` (imported lazily so
    ``import dbhandlers.pg_adapter`` stays cheap and DB-free).
    """

    def __init__(self, engine: Engine, metadata: Optional[MetaData] = None):
        self._engine = engine
        if metadata is None:
            from models.db import Base
            metadata = Base.metadata
        self._metadata = metadata

    def table(self, name: str) -> _Table:
        table = self._metadata.tables.get(name)
        if table is None:
            raise RelationDoesNotExistError(
                f'relation "{name}" does not exist'
            )
        return _Table(self._engine, table)
