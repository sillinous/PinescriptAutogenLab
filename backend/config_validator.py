# backend/config_validator.py

"""
Configuration validation on application startup.

Validates:
- Required environment variables
- Encryption key format
- Broker credentials format
- SMTP configuration (if enabled)
- Database accessibility
- CORS settings
"""

import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path


class ConfigValidator:
    """Validates application configuration before startup."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def validate_all(self, strict: bool = False) -> Dict[str, Any]:
        """
        Validate all configuration.

        Args:
            strict: If True, warnings are treated as errors

        Returns:
            Validation results
        """
        print("\n" + "="*60)
        print("CONFIGURATION VALIDATION")
        print("="*60 + "\n")

        # Run all validations
        self._validate_required_vars()
        self._validate_encryption_key()
        self._validate_webhook_secret()
        self._validate_broker_config()
        self._validate_database()
        self._validate_smtp()
        self._validate_cors()
        self._validate_logging()
        self._validate_paths()

        # Print results
        self._print_results()

        # Determine if validation passed
        has_errors = len(self.errors) > 0
        has_warnings = len(self.warnings) > 0

        if strict and has_warnings:
            has_errors = True

        return {
            'valid': not has_errors,
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info
        }

    def _validate_required_vars(self):
        """Validate required environment variables."""
        required = {
            'WEBHOOK_SECRET': 'Webhook signature validation',
            'ENCRYPTION_KEY': 'Credential encryption'
        }

        for var, purpose in required.items():
            value = os.getenv(var)
            if not value or value == 'your_secret_key_here' or value == 'your_encryption_key_here':
                self.errors.append(
                    f"{var} not set or using default value. Required for: {purpose}"
                )
            else:
                self.info.append(f"✓ {var} is configured")

    def _validate_encryption_key(self):
        """Validate encryption key format."""
        key = os.getenv('ENCRYPTION_KEY')

        if not key or key == 'your_encryption_key_here':
            self.errors.append(
                "ENCRYPTION_KEY not set. Generate with: "
                "python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
            )
            return

        # Check if it's a valid Fernet key (44 characters, base64, ends with =)
        if len(key) == 44 and key.endswith('='):
            try:
                from cryptography.fernet import Fernet
                Fernet(key.encode())
                self.info.append("✓ ENCRYPTION_KEY format is valid (Fernet)")
            except Exception as e:
                self.errors.append(f"ENCRYPTION_KEY is invalid Fernet key: {e}")
        else:
            # Might be a password that will be derived
            if len(key) < 16:
                self.warnings.append(
                    f"ENCRYPTION_KEY is short ({len(key)} chars). "
                    "Recommend using Fernet key (44 chars) or longer password (32+ chars)"
                )
            else:
                self.info.append(f"✓ ENCRYPTION_KEY is set ({len(key)} chars, will be derived)")

    def _validate_webhook_secret(self):
        """Validate webhook secret."""
        secret = os.getenv('WEBHOOK_SECRET')

        if not secret or secret == 'your_secret_key_here':
            self.errors.append("WEBHOOK_SECRET not set. Generate with: python -c \"import secrets; print(secrets.token_hex(32))\"")
            return

        if len(secret) < 32:
            self.warnings.append(
                f"WEBHOOK_SECRET is short ({len(secret)} chars). "
                "Recommend at least 32 characters for security."
            )
        else:
            self.info.append(f"✓ WEBHOOK_SECRET is configured ({len(secret)} chars)")

    def _validate_broker_config(self):
        """Validate broker configuration."""
        # Alpaca
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        paper_trading = os.getenv('ALPACA_PAPER_TRADING', 'true').lower()

        if not api_key or api_key == 'your_alpaca_api_key':
            self.warnings.append("ALPACA_API_KEY not configured. Alpaca integration will not work.")
        elif not api_key.startswith('PK') and not api_key.startswith('AK'):
            self.warnings.append("ALPACA_API_KEY format looks incorrect (should start with PK or AK)")
        else:
            self.info.append(f"✓ ALPACA_API_KEY configured ({'Paper' if paper_trading == 'true' else 'Live'} trading)")

        if not secret_key or secret_key == 'your_alpaca_secret_key':
            self.warnings.append("ALPACA_SECRET_KEY not configured")
        elif len(secret_key) < 32:
            self.warnings.append(f"ALPACA_SECRET_KEY looks too short ({len(secret_key)} chars)")

        if paper_trading != 'true' and paper_trading != 'false':
            self.warnings.append(f"ALPACA_PAPER_TRADING has invalid value '{paper_trading}' (should be 'true' or 'false')")

        if paper_trading == 'false':
            self.warnings.append("[WARN]  LIVE TRADING ENABLED! Make sure this is intentional.")

    def _validate_database(self):
        """Validate database configuration."""
        data_dir = os.getenv('PINELAB_DATA', './data')
        db_path = Path(data_dir)

        if not db_path.exists():
            try:
                db_path.mkdir(parents=True, exist_ok=True)
                self.info.append(f"✓ Created data directory: {data_dir}")
            except Exception as e:
                self.errors.append(f"Cannot create data directory {data_dir}: {e}")
        else:
            self.info.append(f"✓ Data directory exists: {data_dir}")

        # Check write permissions
        test_file = db_path / '.write_test'
        try:
            test_file.write_text('test')
            test_file.unlink()
            self.info.append(f"✓ Data directory is writable")
        except Exception as e:
            self.errors.append(f"Data directory not writable: {e}")

    def _validate_smtp(self):
        """Validate SMTP configuration."""
        smtp_user = os.getenv('SMTP_USER')
        smtp_password = os.getenv('SMTP_PASSWORD')
        smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        smtp_port = os.getenv('SMTP_PORT', '587')

        if not smtp_user or not smtp_password:
            self.info.append("ℹ  SMTP not configured - email notifications disabled (optional)")
            return

        # Validate email format
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_regex, smtp_user):
            self.warnings.append(f"SMTP_USER doesn't look like valid email: {smtp_user}")

        # Validate port
        try:
            port = int(smtp_port)
            if port not in [25, 465, 587, 2525]:
                self.warnings.append(f"SMTP_PORT {port} is unusual (common: 587, 465, 25)")
        except ValueError:
            self.errors.append(f"SMTP_PORT must be a number, got: {smtp_port}")

        self.info.append(f"✓ SMTP configured: {smtp_host}:{smtp_port} (user: {smtp_user})")

        # Test connection (optional, can be slow)
        test_smtp = os.getenv('TEST_SMTP_ON_STARTUP', 'false').lower() == 'true'
        if test_smtp:
            self._test_smtp_connection(smtp_host, int(smtp_port), smtp_user, smtp_password)

    def _test_smtp_connection(self, host: str, port: int, user: str, password: str):
        """Test SMTP connection (optional)."""
        try:
            import smtplib
            with smtplib.SMTP(host, port, timeout=5) as server:
                server.starttls()
                server.login(user, password)
            self.info.append("✓ SMTP connection test successful")
        except Exception as e:
            self.warnings.append(f"SMTP connection test failed: {e}")

    def _validate_cors(self):
        """Validate CORS configuration."""
        cors_origins = os.getenv('CORS_ORIGINS', '')

        if not cors_origins:
            self.warnings.append("CORS_ORIGINS not set - using default (http://localhost:5173)")
            return

        origins = [o.strip() for o in cors_origins.split(',')]

        # Check for wildcards in production
        if '*' in cors_origins:
            self.warnings.append("[WARN]  CORS allows all origins (*) - not recommended for production")

        # Validate origin format
        for origin in origins:
            if origin == '*':
                continue
            if not origin.startswith('http://') and not origin.startswith('https://'):
                self.warnings.append(f"CORS origin '{origin}' should start with http:// or https://")

        self.info.append(f"✓ CORS configured for {len(origins)} origin(s)")

    def _validate_logging(self):
        """Validate logging configuration."""
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()

        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if log_level not in valid_levels:
            self.warnings.append(
                f"LOG_LEVEL '{log_level}' is invalid. "
                f"Valid options: {', '.join(valid_levels)}"
            )
        else:
            self.info.append(f"✓ Logging level: {log_level}")

        if log_level == 'DEBUG':
            self.warnings.append("[WARN]  DEBUG logging enabled - may impact performance and expose sensitive data")

    def _validate_paths(self):
        """Validate path configurations."""
        data_dir = os.getenv('PINELAB_DATA', './data')

        # Check if path is relative or absolute
        if not os.path.isabs(data_dir):
            self.info.append(f"ℹ  Using relative data path: {data_dir}")
        else:
            self.info.append(f"✓ Using absolute data path: {data_dir}")

    def _print_results(self):
        """Print validation results to console."""
        print("\nValidation Results:")
        print("-" * 60)

        if self.errors:
            print(f"\n[ERROR] ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")

        if self.warnings:
            print(f"\n[WARN]  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")

        if self.info:
            print(f"\n✓ INFO ({len(self.info)}):")
            for msg in self.info:
                print(f"  {msg}")

        print("\n" + "="*60)

        if self.errors:
            print("[ERROR] VALIDATION FAILED - Fix errors before starting")
        elif self.warnings:
            print("[WARN]  VALIDATION PASSED WITH WARNINGS")
        else:
            print("[OK] VALIDATION PASSED")

        print("="*60 + "\n")


def validate_config(strict: bool = False) -> bool:
    """
    Validate configuration and return True if valid.

    Args:
        strict: If True, warnings are treated as errors

    Returns:
        True if configuration is valid
    """
    validator = ConfigValidator()
    result = validator.validate_all(strict=strict)
    return result['valid']


def validate_or_exit(strict: bool = False):
    """
    Validate configuration and exit if invalid.

    Args:
        strict: If True, warnings are treated as errors
    """
    if not validate_config(strict=strict):
        print("\n[ERROR] Configuration validation failed. Please fix errors and try again.\n")
        import sys
        sys.exit(1)


# Example usage
if __name__ == "__main__":
    validate_or_exit(strict=False)
