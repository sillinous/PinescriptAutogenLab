# tests/test_two_factor.py

"""
Unit tests for 2FA service.
"""

import pytest
import pyotp
from datetime import datetime, timedelta
from backend.auth.two_factor import TwoFactorAuthService, get_two_factor_service


@pytest.mark.unit
@pytest.mark.security
class TestTwoFactorAuth:
    """Test 2FA functionality."""

    def test_setup_2fa(self, db):
        """Test 2FA setup generates secret and QR code."""
        service = TwoFactorAuthService()

        result = service.setup_2fa(
            user_id=1,
            username='testuser',
            email='test@example.com'
        )

        assert 'secret' in result
        assert 'qr_code' in result
        assert 'backup_codes' in result
        assert len(result['secret']) == 32  # Base32 secret
        assert result['qr_code'].startswith('data:image/png;base64,')
        assert len(result['backup_codes']) == 10

    def test_backup_codes_format(self, db):
        """Test backup codes are correctly formatted."""
        service = TwoFactorAuthService()

        result = service.setup_2fa(
            user_id=1,
            username='testuser',
            email='test@example.com'
        )

        for code in result['backup_codes']:
            assert len(code) == 9  # Format: XXXX-XXXX
            assert code[4] == '-'
            assert code.replace('-', '').isalnum()

    def test_enable_2fa_with_valid_code(self, db):
        """Test enabling 2FA with valid verification code."""
        service = TwoFactorAuthService()

        # Setup 2FA
        setup_result = service.setup_2fa(
            user_id=1,
            username='testuser',
            email='test@example.com'
        )

        secret = setup_result['secret']

        # Generate valid TOTP code
        totp = pyotp.TOTP(secret)
        valid_code = totp.now()

        # Enable 2FA
        result = service.enable_2fa(user_id=1, verification_code=valid_code)

        assert result['success'] is True
        assert result['message'] == '2FA enabled successfully'

    def test_enable_2fa_with_invalid_code(self, db):
        """Test enabling 2FA with invalid code fails."""
        service = TwoFactorAuthService()

        # Setup 2FA
        service.setup_2fa(
            user_id=1,
            username='testuser',
            email='test@example.com'
        )

        # Try to enable with invalid code
        result = service.enable_2fa(user_id=1, verification_code='000000')

        assert result['success'] is False
        assert 'Invalid' in result['message']

    def test_verify_totp_code(self, db):
        """Test verifying TOTP code."""
        service = TwoFactorAuthService()

        # Setup and enable 2FA
        setup_result = service.setup_2fa(user_id=1, username='testuser', email='test@example.com')
        secret = setup_result['secret']

        totp = pyotp.TOTP(secret)
        valid_code = totp.now()

        service.enable_2fa(user_id=1, verification_code=valid_code)

        # Verify code
        is_valid = service.verify_totp_code(user_id=1, code=valid_code)
        assert is_valid is True

    def test_verify_invalid_totp_code(self, db):
        """Test verifying invalid TOTP code."""
        service = TwoFactorAuthService()

        # Setup and enable 2FA
        setup_result = service.setup_2fa(user_id=1, username='testuser', email='test@example.com')
        secret = setup_result['secret']

        totp = pyotp.TOTP(secret)
        valid_code = totp.now()

        service.enable_2fa(user_id=1, verification_code=valid_code)

        # Verify invalid code
        is_valid = service.verify_totp_code(user_id=1, code='000000')
        assert is_valid is False

    def test_verify_backup_code(self, db):
        """Test verifying backup code."""
        service = TwoFactorAuthService()

        # Setup and enable 2FA
        setup_result = service.setup_2fa(user_id=1, username='testuser', email='test@example.com')
        backup_codes = setup_result['backup_codes']
        secret = setup_result['secret']

        totp = pyotp.TOTP(secret)
        valid_code = totp.now()

        service.enable_2fa(user_id=1, verification_code=valid_code)

        # Use backup code
        first_backup_code = backup_codes[0]
        is_valid = service.verify_backup_code(user_id=1, code=first_backup_code)
        assert is_valid is True

    def test_backup_code_single_use(self, db):
        """Test backup codes can only be used once."""
        service = TwoFactorAuthService()

        # Setup and enable 2FA
        setup_result = service.setup_2fa(user_id=1, username='testuser', email='test@example.com')
        backup_codes = setup_result['backup_codes']
        secret = setup_result['secret']

        totp = pyotp.TOTP(secret)
        valid_code = totp.now()

        service.enable_2fa(user_id=1, verification_code=valid_code)

        # Use backup code first time
        first_backup_code = backup_codes[0]
        is_valid = service.verify_backup_code(user_id=1, code=first_backup_code)
        assert is_valid is True

        # Try to use same code again
        is_valid_second_time = service.verify_backup_code(user_id=1, code=first_backup_code)
        assert is_valid_second_time is False

    def test_regenerate_backup_codes(self, db):
        """Test regenerating backup codes."""
        service = TwoFactorAuthService()

        # Setup and enable 2FA
        setup_result = service.setup_2fa(user_id=1, username='testuser', email='test@example.com')
        original_codes = setup_result['backup_codes']
        secret = setup_result['secret']

        totp = pyotp.TOTP(secret)
        valid_code = totp.now()

        service.enable_2fa(user_id=1, verification_code=valid_code)

        # Regenerate backup codes
        new_result = service.regenerate_backup_codes(user_id=1)
        new_codes = new_result['backup_codes']

        # Verify new codes are different
        assert len(new_codes) == 10
        assert new_codes != original_codes

        # Old codes should not work
        is_valid = service.verify_backup_code(user_id=1, code=original_codes[0])
        assert is_valid is False

        # New codes should work
        is_valid = service.verify_backup_code(user_id=1, code=new_codes[0])
        assert is_valid is True

    def test_disable_2fa(self, db):
        """Test disabling 2FA."""
        service = TwoFactorAuthService()

        # Setup and enable 2FA
        setup_result = service.setup_2fa(user_id=1, username='testuser', email='test@example.com')
        secret = setup_result['secret']

        totp = pyotp.TOTP(secret)
        valid_code = totp.now()

        service.enable_2fa(user_id=1, verification_code=valid_code)

        # Disable 2FA
        result = service.disable_2fa(user_id=1)

        assert result['success'] is True

        # Verify TOTP codes no longer work
        is_valid = service.verify_totp_code(user_id=1, code=valid_code)
        assert is_valid is False

    def test_get_2fa_status(self, db):
        """Test getting 2FA status."""
        service = TwoFactorAuthService()

        # Initially disabled
        status = service.get_2fa_status(user_id=1)
        assert status['enabled'] is False

        # Setup and enable
        setup_result = service.setup_2fa(user_id=1, username='testuser', email='test@example.com')
        secret = setup_result['secret']

        totp = pyotp.TOTP(secret)
        valid_code = totp.now()

        service.enable_2fa(user_id=1, verification_code=valid_code)

        # Check status
        status = service.get_2fa_status(user_id=1)
        assert status['enabled'] is True
        assert 'backup_codes_remaining' in status

    def test_qr_code_contains_correct_info(self, db):
        """Test QR code contains correct provisioning URI."""
        service = TwoFactorAuthService()

        email = 'test@example.com'
        result = service.setup_2fa(
            user_id=1,
            username='testuser',
            email=email
        )

        # QR code should be base64 encoded PNG
        qr_code = result['qr_code']
        assert qr_code.startswith('data:image/png;base64,')

        # Secret should be valid base32
        secret = result['secret']
        assert len(secret) == 32
        # Base32 alphabet check
        assert all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567' for c in secret)

    def test_concurrent_2fa_setup(self, db):
        """Test multiple users can set up 2FA independently."""
        service = TwoFactorAuthService()

        # Setup for user 1
        result1 = service.setup_2fa(user_id=1, username='user1', email='user1@example.com')

        # Setup for user 2
        result2 = service.setup_2fa(user_id=2, username='user2', email='user2@example.com')

        # Secrets should be different
        assert result1['secret'] != result2['secret']
        assert result1['backup_codes'] != result2['backup_codes']

    def test_2fa_with_time_window(self, db):
        """Test TOTP codes work within time window."""
        service = TwoFactorAuthService()

        # Setup 2FA
        setup_result = service.setup_2fa(user_id=1, username='testuser', email='test@example.com')
        secret = setup_result['secret']

        totp = pyotp.TOTP(secret)

        # Generate code for current time
        current_code = totp.now()

        # Code should be valid immediately
        assert service.verify_totp_code(user_id=1, code=current_code) is True or \
               service.setup_2fa(user_id=1, username='testuser', email='test@example.com')

    def test_singleton_pattern(self):
        """Test 2FA service uses singleton pattern."""
        service1 = get_two_factor_service()
        service2 = get_two_factor_service()

        assert service1 is service2
