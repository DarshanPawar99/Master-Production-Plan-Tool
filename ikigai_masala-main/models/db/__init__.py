"""SQLAlchemy ORM models for the AlloyDB ``menu_engineering`` database.

These mirror the tool's CURRENT Supabase schema (see scripts/alloydb_setup.sql).
``Base.metadata`` is what the query adapter uses to translate Supabase-style
calls into SQL, so the columns here ARE the contract.
"""

from models.db.app_setting import AppSetting
from models.db.base import Base
from models.db.client import Client
from models.db.menu_history import MenuHistory
from models.db.week_signature import WeekSignature

__all__ = [
    "Base",
    "Client",
    "AppSetting",
    "MenuHistory",
    "WeekSignature",
]
