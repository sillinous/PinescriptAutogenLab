# backend/config.py

import os
from pathlib import Path
from typing import List

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[OK] Loaded environment from {env_path}")
except ImportError:
    print("[WARN] python-dotenv not installed. Using system environment variables only.")


class Config:
    """Application configuration."""

    # Data directory
    DATA_DIR = os.getenv("PINELAB_DATA", "./data")

    # Security secrets
    WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
    REQUIRE_WEBHOOK_SIGNATURE = bool(WEBHOOK_SECRET)
    JWT_SECRET = os.getenv("JWT_SECRET", "")
    ENCRYPTION_SALT = os.getenv("ENCRYPTION_SALT", "")

    # Alpaca configuration
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
    ALPACA_PAPER_TRADING = os.getenv("ALPACA_PAPER_TRADING", "true").lower() == "true"
    ALPACA_BASE_URL = (
        "https://paper-api.alpaca.markets" if ALPACA_PAPER_TRADING
        else "https://api.alpaca.markets"
    )

    # Server configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8080"))
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        warnings = []

        if not cls.WEBHOOK_SECRET:
            warnings.append("[WARN] WEBHOOK_SECRET not set - webhooks will accept any request (INSECURE)")

        if not cls.ALPACA_API_KEY or not cls.ALPACA_SECRET_KEY:
            warnings.append("[WARN] Alpaca credentials not configured - equities trading disabled")

        return warnings


# Print config warnings on import
config_warnings = Config.validate()
if config_warnings:
    print("\n" + "\n".join(config_warnings) + "\n")
