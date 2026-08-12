"""Shared constants for the AlloyDB connection layer."""

# AlloyDB database that holds the menu-engineering tables
# (clients, app_settings, menu_history, week_signatures).
# Overridable per-deployment via the secrets file / DB_NAME env var; this is
# only the default used when the secret does not name a database.
DB_NAME = "menu_engineering"
