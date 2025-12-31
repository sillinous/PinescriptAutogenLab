# backend/security.py

import hmac
import hashlib
from typing import Optional
from backend.config import Config


def verify_webhook_signature(payload: bytes, signature: Optional[str]) -> bool:
    """
    Verify HMAC-SHA256 signature for TradingView webhooks.

    TradingView should send signature in X-Signature header as hex(hmac_sha256(secret, raw_body))
    """
    if not Config.REQUIRE_WEBHOOK_SIGNATURE:
        return True  # Skip validation if no secret configured

    if not signature:
        return False

    try:
        expected_signature = hmac.new(
            Config.WEBHOOK_SECRET.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(signature, expected_signature)
    except Exception as e:
        print(f"[ERROR] Signature validation error: {e}")
        return False


def generate_signature_example(payload: str) -> str:
    """Generate example signature for testing."""
    if not Config.WEBHOOK_SECRET:
        return "NO_SECRET_CONFIGURED"

    return hmac.new(
        Config.WEBHOOK_SECRET.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
