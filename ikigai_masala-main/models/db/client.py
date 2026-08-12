"""clients table — per-client config in JSONB + plain per-client columns.

Matches the tool's CURRENT Supabase schema exactly (see scripts/setup_all.sql
and scripts/alloydb_setup.sql). The whole cuisine config lives in ``counters``;
the remaining columns are plain per-client attributes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from sqlalchemy import Boolean, DateTime, Integer, String, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from models.db.base import Base

# JSONB on Postgres/AlloyDB, plain JSON on sqlite (so unit tests can run
# without a Postgres). Behaviour is identical for the dict/list values we store.
_JSON = JSONB().with_variant(JSON(), "sqlite")


class Client(Base):
    __tablename__ = "clients"

    name: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1"), default=1,
    )
    counters: Mapped[List[Any]] = mapped_column(
        _JSON, nullable=False, server_default=text("'[]'::jsonb"), default=list,
    )
    city: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    serve_weekends: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false"), default=False,
    )
    item_cooldown_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    source_pools: Mapped[Optional[Any]] = mapped_column(_JSON, nullable=True)
    working_days: Mapped[Optional[Any]] = mapped_column(_JSON, nullable=True)
    is_launch_site: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false"), default=False,
    )
    shared_categories: Mapped[Optional[Any]] = mapped_column(_JSON, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )
