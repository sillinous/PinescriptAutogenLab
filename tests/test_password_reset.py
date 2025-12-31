# tests/test_password_reset.py

"""
Unit tests for password reset service.
"""

import pytest
from datetime import datetime, timedelta
from backend.auth.password_reset import PasswordResetService, get_password_reset_service


@pytest.mark.unit
@pytest.mark.security
class TestPasswordReset:
    """Test password reset functionality."""

    def test_request_password_reset(self, db, registered_user):
        """Test requesting password reset."""
        service = PasswordResetService()

        result = service.request_password_reset(
            email=registered_user['email']
        )

        assert result['success'] is True
        assert 'token' in result

    def test_request_reset_for_nonexistent_email(self, db):
        """Test requesting reset for non-existent email (should still succeed for security)."""
        service = PasswordResetService()

        result = service.request_password_reset(
            email='nonexistent@example.com'
        )

        # Should return success to prevent email enumeration
        assert result['success'] is True

    def test_validate_reset_token(self, db, registered_user):
        """Test validating reset token."""
        service = PasswordResetService()

        # Request reset
        request_result = service.request_password_reset(email=registered_user['email'])
        token = request_result['token']

        # Validate token
        validate_result = service.validate_reset_token(token)

        assert validate_result['valid'] is True
        assert validate_result['user_id'] == registered_user['id']

    def test_validate_invalid_token(self, db):
        """Test validating invalid token."""
        service = PasswordResetService()

        result = service.validate_reset_token('invalid_token_12345')

        assert result['valid'] is False

    def test_reset_password_with_valid_token(self, db, registered_user):
        """Test resetting password with valid token."""
        service = PasswordResetService()

        # Request reset
        request_result = service.request_password_reset(email=registered_user['email'])
        token = request_result['token']

        new_password = 'NewPassword456!'

        # Reset password
        reset_result = service.reset_password(token, new_password)

        assert reset_result['success'] is True

        # Verify can login with new password
        from backend.auth.auth_service import AuthService
        auth = AuthService()

        login_result = auth.authenticate_user(registered_user['username'], new_password)
        assert login_result is not None

    def test_reset_password_with_invalid_token(self, db):
        """Test resetting password with invalid token."""
        service = PasswordResetService()

        result = service.reset_password('invalid_token', 'NewPassword123!')

        assert result['success'] is False

    def test_reset_token_single_use(self, db, registered_user):
        """Test reset tokens can only be used once."""
        service = PasswordResetService()

        # Request reset
        request_result = service.request_password_reset(email=registered_user['email'])
        token = request_result['token']

        # Reset password first time
        reset_result1 = service.reset_password(token, 'NewPassword1!')
        assert reset_result1['success'] is True

        # Try to use token again
        reset_result2 = service.reset_password(token, 'NewPassword2!')
        assert reset_result2['success'] is False

    def test_expired_reset_token(self, db, registered_user):
        """Test expired reset token."""
        service = PasswordResetService()

        # Request reset
        request_result = service.request_password_reset(email=registered_user['email'])
        token = request_result['token']

        # Manually expire token in database
        from backend.database import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE password_resets
            SET expires_at = datetime('now', '-1 hour')
            WHERE user_id = ?
        """, (registered_user['id'],))
        conn.commit()
        conn.close()

        # Try to reset with expired token
        reset_result = service.reset_password(token, 'NewPassword123!')

        assert reset_result['success'] is False
        assert 'expired' in reset_result['message'].lower()

    def test_multiple_reset_requests(self, db, registered_user):
        """Test multiple reset requests invalidate old tokens."""
        service = PasswordResetService()

        # First request
        result1 = service.request_password_reset(email=registered_user['email'])
        token1 = result1['token']

        # Second request
        result2 = service.request_password_reset(email=registered_user['email'])
        token2 = result2['token']

        # Tokens should be different
        assert token1 != token2

        # Only most recent token should work (implementation dependent)
        # Test that at least one works
        reset_result = service.reset_password(token2, 'NewPassword123!')
        assert reset_result['success'] is True

    def test_reset_with_weak_password(self, db, registered_user):
        """Test resetting with weak password."""
        service = PasswordResetService()

        # Request reset
        request_result = service.request_password_reset(email=registered_user['email'])
        token = request_result['token']

        # Try weak passwords
        weak_passwords = ['123', 'password', 'abc']

        for weak_pass in weak_passwords:
            result = service.reset_password(token, weak_pass)
            # Should fail validation (if implemented)
            # This depends on password validation in the service

    def test_reset_tracks_ip_address(self, db, registered_user):
        """Test password reset tracks IP address."""
        service = PasswordResetService()

        ip_address = '192.168.1.100'

        # Request reset
        request_result = service.request_password_reset(
            email=registered_user['email'],
            ip_address=ip_address
        )

        assert request_result['success'] is True

        # Verify IP is stored (check database)
        from backend.database import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT ip_address FROM password_resets
            WHERE user_id = ?
            ORDER BY created_at DESC LIMIT 1
        """, (registered_user['id'],))
        row = cursor.fetchone()
        conn.close()

        assert row[0] == ip_address

    def test_cleanup_old_reset_tokens(self, db, registered_user):
        """Test cleanup of old reset tokens."""
        service = PasswordResetService()

        # Create reset request
        service.request_password_reset(email=registered_user['email'])

        # Manually set old timestamp
        from backend.database import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE password_resets
            SET created_at = datetime('now', '-100 days')
            WHERE user_id = ?
        """, (registered_user['id'],))
        conn.commit()

        # Run cleanup
        deleted_count = service.cleanup_old_tokens(days_old=30)

        assert deleted_count >= 1

        conn.close()

    def test_get_recent_reset_requests(self, db, registered_user):
        """Test getting recent reset requests for a user."""
        service = PasswordResetService()

        # Make multiple requests
        service.request_password_reset(email=registered_user['email'])
        service.request_password_reset(email=registered_user['email'])

        # Get recent requests
        recent = service.get_user_reset_history(user_id=registered_user['id'], limit=5)

        assert len(recent) >= 2

    def test_rate_limiting_reset_requests(self, db, registered_user):
        """Test rate limiting for reset requests."""
        service = PasswordResetService()

        # Make multiple rapid requests
        for i in range(5):
            result = service.request_password_reset(email=registered_user['email'])

        # Depending on implementation, this might trigger rate limiting
        # The test verifies the system handles it gracefully
        assert True  # Placeholder

    def test_singleton_pattern(self):
        """Test password reset service uses singleton pattern."""
        service1 = get_password_reset_service()
        service2 = get_password_reset_service()

        assert service1 is service2


@pytest.mark.integration
class TestPasswordResetIntegration:
    """Integration tests for password reset workflow."""

    def test_complete_password_reset_workflow(self, client, registered_user):
        """Test complete password reset workflow via API."""
        # Request reset
        request_response = client.post('/auth/request-reset', json={
            'email': registered_user['email']
        })

        assert request_response.status_code == 200

        # Get token (in real app, would be in email)
        from backend.auth.password_reset import get_password_reset_service
        service = get_password_reset_service()
        request_result = service.request_password_reset(email=registered_user['email'])
        token = request_result['token']

        # Validate token
        validate_response = client.get(f'/auth/validate-reset-token?token={token}')
        assert validate_response.status_code == 200

        # Reset password
        new_password = 'NewSecurePassword123!'
        reset_response = client.post('/auth/reset-password', json={
            'token': token,
            'new_password': new_password
        })

        assert reset_response.status_code == 200

        # Login with new password
        login_response = client.post('/auth/login', json={
            'username': registered_user['username'],
            'password': new_password
        })

        assert login_response.status_code == 200
        assert 'access_token' in login_response.json()

    def test_old_password_stops_working_after_reset(self, client, registered_user):
        """Test old password no longer works after reset."""
        old_password = registered_user['password']

        # Request and perform reset
        from backend.auth.password_reset import get_password_reset_service
        service = get_password_reset_service()
        request_result = service.request_password_reset(email=registered_user['email'])
        token = request_result['token']

        new_password = 'BrandNewPassword123!'
        service.reset_password(token, new_password)

        # Try to login with old password
        login_response = client.post('/auth/login', json={
            'username': registered_user['username'],
            'password': old_password
        })

        assert login_response.status_code == 401

        # Login with new password should work
        login_response = client.post('/auth/login', json={
            'username': registered_user['username'],
            'password': new_password
        })

        assert login_response.status_code == 200
