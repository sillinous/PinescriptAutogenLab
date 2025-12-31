# tests/test_config_validator.py

"""
Unit tests for the configuration validator.
"""

import pytest
import os
from backend.config_validator import ConfigValidator

@pytest.fixture
def validator():
    """Fixture for a ConfigValidator instance."""
    return ConfigValidator()

@pytest.mark.unit
class TestConfigValidator:
    """Test the configuration validator."""

    def test_missing_required_vars(self, validator, monkeypatch):
        """Test that missing required environment variables are detected."""
        monkeypatch.delenv("WEBHOOK_SECRET", raising=False)
        monkeypatch.delenv("ENCRYPTION_KEY", raising=False)

        results = validator.validate_all()

        assert not results['valid']
        assert len(results['errors']) >= 2
        assert any("WEBHOOK_SECRET not set" in e for e in results['errors'])
        assert any("ENCRYPTION_KEY not set" in e for e in results['errors'])

    def test_short_webhook_secret(self, validator, monkeypatch):
        """Test that a short webhook secret generates a warning."""
        monkeypatch.setenv("WEBHOOK_SECRET", "short")
        monkeypatch.setenv("ENCRYPTION_KEY", "a_valid_key_that_is_long_enough_for_the_test")

        results = validator.validate_all()

        assert results['valid']
        assert len(results['warnings']) >= 1
        assert any("WEBHOOK_SECRET is short" in w for w in results['warnings'])

    def test_invalid_encryption_key(self, validator, monkeypatch):
        """Test that an invalid encryption key is detected."""
        monkeypatch.setenv("WEBHOOK_SECRET", "a_valid_secret_that_is_long_enough_for_the_test")
        monkeypatch.setenv("ENCRYPTION_KEY", "invalid_key")

        results = validator.validate_all()

        assert results['valid']
        assert len(results['warnings']) >= 1
        assert any("ENCRYPTION_KEY is short" in w for w in results['warnings'])

    def test_valid_config(self, validator, monkeypatch):
        """Test that a valid configuration passes."""
        monkeypatch.setenv("WEBHOOK_SECRET", "a_super_long_and_secure_webhook_secret_for_testing")
        monkeypatch.setenv("ENCRYPTION_KEY", "a_super_long_and_secure_encryption_key_for_testing")
        monkeypatch.setenv("ALPACA_API_KEY", "PKA_VALID_KEY")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "a_super_long_and_secure_alpaca_secret_key")

        results = validator.validate_all()

        assert results['valid']
        assert len(results['errors']) == 0
        assert len(results['warnings']) == 0

    def test_database_validation(self, validator, monkeypatch, tmp_path):
        """Test database path validation."""
        data_dir = tmp_path / "data"
        monkeypatch.setenv("PINELAB_DATA", str(data_dir))

        results = validator.validate_all()

        assert any("Created data directory" in i for i in results['info'])
        assert any("Data directory is writable" in i for i in results['info'])

    def test_smtp_validation(self, validator, monkeypatch):
        """Test SMTP configuration validation."""
        monkeypatch.setenv("SMTP_USER", "invalid-email")
        monkeypatch.setenv("SMTP_PASSWORD", "password")
        monkeypatch.setenv("SMTP_PORT", "9999")

        results = validator.validate_all()

        assert any("SMTP_USER doesn't look like valid email" in w for w in results['warnings'])
        assert any("SMTP_PORT 9999 is unusual" in w for w in results['warnings'])

    def test_cors_validation(self, validator, monkeypatch):
        """Test CORS configuration validation."""
        monkeypatch.setenv("CORS_ORIGINS", "*")

        results = validator.validate_all()

        assert any("CORS allows all origins (*)" in w for w in results['warnings'])
