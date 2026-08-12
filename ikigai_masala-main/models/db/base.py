"""Base model for SQLAlchemy models."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models.

    ``Base.metadata`` is the single source of table definitions the AlloyDB
    query adapter (``dbhandlers.pg_adapter``) reflects on to translate the
    Supabase-style ``.table(...).select()...`` calls into SQL.
    """
