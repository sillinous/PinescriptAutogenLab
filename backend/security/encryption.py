# backend/security/encryption.py

"""
Encryption utilities for sensitive data storage.

Provides AES-256 encryption for API keys, broker credentials, and other sensitive data.
Uses Fernet (symmetric encryption) from cryptography library.
"""

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import os
from typing import Optional

# Flag to indicate if encryption is available
ENCRYPTION_AVAILABLE = True


class EncryptionService:
    """Service for encrypting/decrypting sensitive data."""

    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize encryption service.

        Args:
            master_key: Master encryption key (from environment)
                       If not provided, will use ENCRYPTION_KEY from env
        """
        key = master_key or os.getenv("ENCRYPTION_KEY")

        if not key:
            raise ValueError(
                "ENCRYPTION_KEY not set. Generate with: "
                "python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
            )

        # Derive encryption key from master key
        if len(key) == 44 and key.endswith('='):
            # Already a valid Fernet key
            self.cipher = Fernet(key.encode())
        else:
            # Derive from password using PBKDF2
            # Salt from environment variable for production security
            salt_hex = os.getenv("ENCRYPTION_SALT", "")
            if not salt_hex:
                raise ValueError(
                    "ENCRYPTION_SALT not set. Generate with: "
                    "python -c \"import secrets; print(secrets.token_hex(16))\""
                )
            salt = bytes.fromhex(salt_hex)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            key_bytes = base64.urlsafe_b64encode(kdf.derive(key.encode()))
            self.cipher = Fernet(key_bytes)

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a string.

        Args:
            plaintext: String to encrypt

        Returns:
            Base64-encoded encrypted string
        """
        if not plaintext:
            return ""

        encrypted = self.cipher.encrypt(plaintext.encode())
        return encrypted.decode()

    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt a string.

        Args:
            ciphertext: Encrypted string (base64-encoded)

        Returns:
            Decrypted plaintext string
        """
        if not ciphertext:
            return ""

        try:
            decrypted = self.cipher.decrypt(ciphertext.encode())
            return decrypted.decode()
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")


# Global encryption service instance
_encryption_service: Optional[EncryptionService] = None


def get_encryption_service() -> EncryptionService:
    """Get or create global encryption service instance."""
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = EncryptionService()
    return _encryption_service


def encrypt_credential(plaintext: str) -> str:
    """Encrypt a credential string."""
    service = get_encryption_service()
    return service.encrypt(plaintext)


def decrypt_credential(ciphertext: str) -> str:
    """Decrypt a credential string."""
    service = get_encryption_service()
    return service.decrypt(ciphertext)


# Example usage and setup instructions
"""
Setup Instructions:
===================

1. Generate encryption key:
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

2. Add to .env:
   ENCRYPTION_KEY=<generated_key>

3. Usage in code:
   from backend.security.encryption import encrypt_credential, decrypt_credential

   # Encrypt before storing
   encrypted_api_key = encrypt_credential(api_key)

   # Decrypt when needed
   api_key = decrypt_credential(encrypted_api_key)

Security Notes:
===============
- ENCRYPTION_KEY must be kept secret
- Never commit ENCRYPTION_KEY to version control
- In production, use a key management service (AWS KMS, HashiCorp Vault)
- Rotate encryption keys periodically
- Store encryption key separately from database
"""
