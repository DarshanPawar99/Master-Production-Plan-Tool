"""week_signatures — one row per saved week plan (week-level cooldowns)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import BigInteger, Date, DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column

from models.db.base import Base


class WeekSignature(Base):
    __tablename__ = "week_signatures"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    client_name: Mapped[str] = mapped_column(
        String, ForeignKey("clients.name", ondelete="CASCADE"), nullable=False,
    )
    week_start: Mapped[date] = mapped_column(Date, nullable=False)
    week_signature: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )
