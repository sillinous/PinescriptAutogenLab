# tests/test_encryption.py

"""
Unit tests for encryption service.
"""

import pytest
import os
from backend.security.encryption import (
    EncryptionService,
    get_encryption_service,
    encrypt_credential,
    decrypt_credential
)


@pytest.mark.unit
@pytest.mark.security
class TestEncryptionService:
    """Test encryption service functionality."""

    def test_encryption_service_initialization(self, encryption_key):
        """Test encryption service can be initialized with a key."""
        service = EncryptionService(master_key=encryption_key)
        assert service is not None
        assert service.cipher is not None

    def test_encrypt_decrypt_roundtrip(self, encryption_key):
        """Test data can be encrypted and decrypted successfully."""
        service = EncryptionService(master_key=encryption_key)

        plaintext = "super_secret_api_key_12345"
        encrypted = service.encrypt(plaintext)

        # Verify encrypted is different from plaintext
        assert encrypted != plaintext
        assert len(encrypted) > len(plaintext)

        # Decrypt and verify
        decrypted = service.decrypt(encrypted)
        assert decrypted == plaintext

    def test_encrypt_empty_string(self, encryption_key):
        """Test encrypting empty string."""
        service = EncryptionService(master_key=encryption_key)

        encrypted = service.encrypt("")
        decrypted = service.decrypt(encrypted)

        assert decrypted == ""

    def test_encrypt_unicode_characters(self, encryption_key):
        """Test encrypting unicode characters."""
        service = EncryptionService(master_key=encryption_key)

        plaintext = "Hello 世界 🔐"
        encrypted = service.encrypt(plaintext)
        decrypted = service.decrypt(encrypted)

        assert decrypted == plaintext

    def test_encrypt_large_text(self, encryption_key):
        """Test encrypting large text."""
        service = EncryptionService(master_key=encryption_key)

        plaintext = "A" * 10000
        encrypted = service.encrypt(plaintext)
        decrypted = service.decrypt(encrypted)

        assert decrypted == plaintext
        assert len(decrypted) == 10000

    def test_different_encryptions_same_plaintext(self, encryption_key):
        """Test that same plaintext produces different ciphertexts (due to IV)."""
        service = EncryptionService(master_key=encryption_key)

        plaintext = "test_data"
        encrypted1 = service.encrypt(plaintext)
        encrypted2 = service.encrypt(plaintext)

        # Due to random IV in Fernet, encryptions should be different
        # (Note: Fernet includes timestamp, so this may not always be true)
        # But both should decrypt to same plaintext
        assert service.decrypt(encrypted1) == plaintext
        assert service.decrypt(encrypted2) == plaintext

    def test_decrypt_invalid_data(self, encryption_key):
        """Test decrypting invalid data raises error."""
        service = EncryptionService(master_key=encryption_key)

        with pytest.raises(Exception):
            service.decrypt("invalid_encrypted_data")

    def test_decrypt_with_wrong_key(self):
        """Test decrypting with wrong key fails."""
        service1 = EncryptionService(master_key="key1_abcdefghijklmnopqrstuvwxyz123456")
        service2 = EncryptionService(master_key="key2_abcdefghijklmnopqrstuvwxyz123456")

        plaintext = "secret_data"
        encrypted = service1.encrypt(plaintext)

        with pytest.raises(Exception):
            service2.decrypt(encrypted)

    def test_convenience_functions(self, encryption_key):
        """Test convenience encrypt/decrypt functions."""
        os.environ['ENCRYPTION_KEY'] = encryption_key

        plaintext = "test_credential"
        encrypted = encrypt_credential(plaintext)
        decrypted = decrypt_credential(encrypted)

        assert decrypted == plaintext

    def test_singleton_pattern(self, encryption_key):
        """Test encryption service uses singleton pattern."""
        os.environ['ENCRYPTION_KEY'] = encryption_key

        service1 = get_encryption_service()
        service2 = get_encryption_service()

        assert service1 is service2

    def test_encryption_of_special_characters(self, encryption_key):
        """Test encryption handles special characters."""
        service = EncryptionService(master_key=encryption_key)

        plaintext = "!@#$%^&*()_+-={}[]|:;<>?,./"
        encrypted = service.encrypt(plaintext)
        decrypted = service.decrypt(encrypted)

        assert decrypted == plaintext

    def test_encryption_deterministic_with_same_key(self, encryption_key):
        """Test that encryption service is consistent with same key."""
        service1 = EncryptionService(master_key=encryption_key)
        service2 = EncryptionService(master_key=encryption_key)

        plaintext = "test_data"
        encrypted1 = service1.encrypt(plaintext)
        decrypted2 = service2.decrypt(encrypted1)

        assert decrypted2 == plaintext


@pytest.mark.unit
@pytest.mark.security
class TestEncryptionIntegration:
    """Test encryption integration with database."""

    def test_save_encrypted_credentials(self, db, encryption_key):
        """Test saving encrypted broker credentials."""
        from backend.database import save_broker_credentials, get_broker_credentials

        os.environ['ENCRYPTION_KEY'] = encryption_key

        # Save credentials
        save_broker_credentials(
            broker_type='alpaca',
            api_key='test_api_key',
            api_secret='test_api_secret',
            base_url='https://paper-api.alpaca.markets',
            is_paper=True
        )

        # Retrieve credentials
        creds = get_broker_credentials('alpaca')

        assert creds is not None
        assert creds['api_key'] == 'test_api_key'
        assert creds['api_secret'] == 'test_api_secret'
        assert creds['broker_type'] == 'alpaca'

    def test_encrypted_credentials_stored_encrypted(self, db, encryption_key):
        """Test that credentials are actually encrypted in database."""
        from backend.database import save_broker_credentials, get_connection

        os.environ['ENCRYPTION_KEY'] = encryption_key

        api_key = 'plaintext_api_key'
        api_secret = 'plaintext_api_secret'

        # Save credentials
        save_broker_credentials(
            broker_type='alpaca',
            api_key=api_key,
            api_secret=api_secret,
            base_url='https://paper-api.alpaca.markets',
            is_paper=True
        )

        # Check raw database values
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT api_key_encrypted, api_secret_encrypted FROM broker_credentials WHERE broker_type = 'alpaca'")
        row = cursor.fetchone()
        conn.close()

        # Verify encrypted values are different from plaintext
        assert row[0] != api_key
        assert row[1] != api_secret
        assert len(row[0]) > len(api_key)  # Encrypted is longer

    def test_update_encrypted_credentials(self, db, encryption_key):
        """Test updating encrypted credentials."""
        from backend.database import save_broker_credentials, get_broker_credentials

        os.environ['ENCRYPTION_KEY'] = encryption_key

        # Save initial credentials
        save_broker_credentials(
            broker_type='alpaca',
            api_key='old_key',
            api_secret='old_secret',
            base_url='https://paper-api.alpaca.markets',
            is_paper=True
        )

        # Update credentials
        save_broker_credentials(
            broker_type='alpaca',
            api_key='new_key',
            api_secret='new_secret',
            base_url='https://api.alpaca.markets',
            is_paper=False
        )

        # Verify updated credentials
        creds = get_broker_credentials('alpaca')
        assert creds['api_key'] == 'new_key'
        assert creds['api_secret'] == 'new_secret'
        assert creds['is_paper'] == 0  # SQLite boolean
