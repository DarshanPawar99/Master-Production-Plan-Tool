"""Tests for api.config.validate_required_env (AlloyDB connection sources)."""

import pytest

from api.config import validate_required_env

# The three self-sufficient connection sources.
_CONN_ENV = ("DATABASE_URL", "EMULATE_LOCAL", "GCP_PROJECT", "GOOGLE_CLOUD_PROJECT")


def _clear(monkeypatch):
    for name in _CONN_ENV:
        monkeypatch.delenv(name, raising=False)


class TestValidateRequiredEnv:
    def test_raises_when_no_source_configured(self, monkeypatch):
        _clear(monkeypatch)
        with pytest.raises(RuntimeError) as exc:
            validate_required_env()
        msg = str(exc.value)
        assert "DATABASE_URL" in msg
        assert "EMULATE_LOCAL" in msg
        assert "GCP_PROJECT" in msg

    def test_database_url_is_sufficient(self, monkeypatch):
        _clear(monkeypatch)
        monkeypatch.setenv("DATABASE_URL", "postgresql+pg8000://u:p@h:5432/db")
        validate_required_env()  # no raise

    def test_emulate_local_is_sufficient(self, monkeypatch):
        _clear(monkeypatch)
        monkeypatch.setenv("EMULATE_LOCAL", "True")
        validate_required_env()  # no raise

    def test_gcp_project_is_sufficient(self, monkeypatch):
        _clear(monkeypatch)
        monkeypatch.setenv("GCP_PROJECT", "my-project")
        validate_required_env()  # no raise

    def test_whitespace_only_does_not_count(self, monkeypatch):
        _clear(monkeypatch)
        monkeypatch.setenv("DATABASE_URL", "   ")
        with pytest.raises(RuntimeError):
            validate_required_env()

    def test_emulate_local_false_is_not_a_source(self, monkeypatch):
        _clear(monkeypatch)
        monkeypatch.setenv("EMULATE_LOCAL", "False")
        with pytest.raises(RuntimeError):
            validate_required_env()
