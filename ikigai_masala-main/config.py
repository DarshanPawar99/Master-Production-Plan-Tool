"""Runtime configuration for the AlloyDB connection (DB bits only).

Mirrors op-mis-backend's ``config.py``. Three ways to source the DB
connection, tried in this order by ``utils.secrets_manager_helper``:

  1. ``DATABASE_URL`` env var  — a full SQLAlchemy URL (tests / one-off scripts)
  2. Local mode (``EMULATE_LOCAL=True``) — a YAML secrets file, else ``DB_*`` env
  3. Cloud (``EMULATE_LOCAL=False``) — GCP Secret Manager secret ``db-connection-config``

Nothing here connects to anything; it just resolves *where the credentials
come from*.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from utils.constants import DB_NAME

# ``EMULATE_LOCAL=True`` skips Secret Manager and reads the local YAML / env.
EMULATE_LOCAL = os.getenv("EMULATE_LOCAL", "False").lower() in ("1", "true", "yes")

# Path to the local YAML secrets file (git-ignored). See secrets.example.yaml.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SECRETS_YAML_PATH = os.getenv(
    "SECRETS_YAML", os.path.join(_REPO_ROOT, "secrets.yaml")
)

GCP_PROJECT: Optional[str] = None

# A full DATABASE_URL bypasses Secret Manager entirely, so there is no need to
# probe for a GCP project (and no reason to emit a warning when the google-cloud
# libs are absent).
_HAS_DATABASE_URL = bool(os.getenv("DATABASE_URL", "").strip())

if EMULATE_LOCAL or _HAS_DATABASE_URL:
    GCP_PROJECT = os.getenv("GCP_PROJECT", "omnipulse-demo")
else:
    GCP_PROJECT = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not GCP_PROJECT:
        try:
            from google.cloud import storage

            GCP_PROJECT = storage.Client().project
        except Exception as exc:  # pragma: no cover — cloud metadata / ADC
            logging.warning(
                "Failed to determine GCP project: %s. "
                "Set GCP_PROJECT when not in local mode.",
                exc,
            )
            GCP_PROJECT = None

# Fallback local connection params (used when EMULATE_LOCAL and no YAML file).
LOCAL_DB_CONFIG = {
    "user": os.getenv("DB_USER", "postgres"),
    "pass": os.getenv("DB_PASS", "postgres"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "name": os.getenv("DB_NAME", DB_NAME),
}
