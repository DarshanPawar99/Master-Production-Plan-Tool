"""menu_history — one row per (client, service_date); the day's whole menu
lives in the ``menu`` JSONB column ({slot: item_base}, or nested
{counter: {slot: item_base}} for multi-counter clients).

Matches the tool's CURRENT schema: the primary key is (client_name,
service_date), NOT a surrogate id, and there is no per-dish row.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from sqlalchemy import Date, DateTime, ForeignKey, String, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from models.db.base import Base

_JSON = JSONB().with_variant(JSON(), "sqlite")


class MenuHistory(Base):
    __tablename__ = "menu_history"

    client_name: Mapped[str] = mapped_column(
        String, ForeignKey("clients.name", ondelete="CASCADE"), primary_key=True,
    )
    service_date: Mapped[date] = mapped_column(Date, primary_key=True)
    menu: Mapped[Any] = mapped_column(
        _JSON, nullable=False, server_default=text("'{}'::jsonb"), default=dict,
    )
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )
