"""Secret Manager / secrets helpers for the AlloyDB connection.

Mirrors op-mis-backend's helper, with one addition: in local mode the DB
config is read from a **YAML** file (``secrets.yaml``) so you can keep the
connection details in one readable place instead of a wall of env vars.

Resolution order for the connection string:

  1. ``DATABASE_URL`` env var (full SQLAlchemy URL)  → used verbatim
  2. ``EMULATE_LOCAL=True`` → ``secrets.yaml`` if present, else ``LOCAL_DB_CONFIG``
  3. otherwise → GCP Secret Manager secret ``db-connection-config`` (JSON)

The DB config dict shape is the same everywhere::

    {"user": ..., "pass": ..., "host": ..., "port": ..., "name": ...}
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional

from config import EMULATE_LOCAL, GCP_PROJECT, LOCAL_DB_CONFIG, SECRETS_YAML_PATH
from utils.constants import DB_NAME

try:
    from google.cloud import secretmanager
except ImportError:  # pragma: no cover — optional at import for unit tests
    secretmanager = None  # type: ignore

try:
    import yaml
except ImportError:  # pragma: no cover — PyYAML optional if not using the YAML path
    yaml = None  # type: ignore

logger = logging.getLogger(__name__)


class SecretsManagerHelper:
    _db_config: Optional[Dict] = None
    _client = None

    @classmethod
    def _get_client(cls):
        if cls._client is None and not EMULATE_LOCAL:
            if secretmanager is None:
                raise RuntimeError(
                    "google-cloud-secret-manager is required when "
                    "EMULATE_LOCAL is False"
                )
            cls._client = secretmanager.SecretManagerServiceClient()
        return cls._client

    @classmethod
    def reset_for_tests(cls) -> None:
        """Drop cached config/client so the next call re-resolves."""
        cls._db_config = None
        cls._client = None

    # -- YAML (local) ------------------------------------------------------

    @staticmethod
    def _load_yaml_config() -> Optional[Dict]:
        """Return the DB config dict from ``secrets.yaml`` if it exists.

        Accepts either a top-level mapping or a ``db:`` sub-mapping::

            db:
              user: postgres
              pass: postgres
              host: 127.0.0.1
              port: 5432
              name: menu_engineering
        """
        path = SECRETS_YAML_PATH
        if not path or not os.path.exists(path):
            return None
        if yaml is None:  # pragma: no cover
            logger.warning(
                "secrets.yaml present at %s but PyYAML is not installed; "
                "falling back to env / LOCAL_DB_CONFIG.", path,
            )
            return None
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        cfg = data.get("db", data) if isinstance(data, dict) else {}
        if not cfg:
            return None
        return {
            "user": cfg.get("user", LOCAL_DB_CONFIG["user"]),
            "pass": cfg.get("pass", LOCAL_DB_CONFIG["pass"]),
            "host": cfg.get("host", LOCAL_DB_CONFIG["host"]),
            "port": int(cfg.get("port", LOCAL_DB_CONFIG["port"])),
            "name": cfg.get("name", LOCAL_DB_CONFIG["name"]),
        }

    # -- config resolution -------------------------------------------------

    @staticmethod
    def get_db_config() -> Dict:
        """Return DB connection params from YAML / env / Secret Manager."""
        if EMULATE_LOCAL:
            yaml_cfg = SecretsManagerHelper._load_yaml_config()
            return yaml_cfg if yaml_cfg is not None else LOCAL_DB_CONFIG

        if SecretsManagerHelper._db_config is None:
            raw = SecretsManagerHelper._get_secret("db-connection-config")
            SecretsManagerHelper._db_config = json.loads(raw)
        return SecretsManagerHelper._db_config

    @staticmethod
    def _get_secret(secret_id: str) -> str:
        if EMULATE_LOCAL:
            if secret_id == "db-connection-config":
                return json.dumps(SecretsManagerHelper.get_db_config())
            return f"fake-{secret_id}"

        if not GCP_PROJECT:
            raise ValueError("GCP_PROJECT is not set; cannot read Secret Manager")

        secret_name = f"projects/{GCP_PROJECT}/secrets/{secret_id}/versions/latest"
        client = SecretsManagerHelper._get_client()
        response = client.access_secret_version(request={"name": secret_name})
        return response.payload.data.decode("UTF-8")

    @staticmethod
    def get_db_connection_string() -> str:
        """AlloyDB (pg8000) SQLAlchemy URL, same shape as op-mis-backend."""
        # Optional full URL override (useful for tests / one-off scripts).
        url = os.getenv("DATABASE_URL", "").strip()
        if url:
            return url

        db_config = SecretsManagerHelper.get_db_config()
        name = db_config.get("name", DB_NAME)
        connection_string = (
            f"postgresql+pg8000://{db_config['user']}:{db_config['pass']}"
            f"@{db_config['host']}:{db_config['port']}/{name}"
        )
        logger.debug("Generated AlloyDB connection string for %s", name)
        return connection_string
