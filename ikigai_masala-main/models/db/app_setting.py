"""app_settings table — global key/value JSON settings."""

from __future__ import annotations

from typing import Any

from sqlalchemy import String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from models.db.base import Base

_JSON = JSONB().with_variant(JSON(), "sqlite")


class AppSetting(Base):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[Any] = mapped_column(_JSON, nullable=False)
