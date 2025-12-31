# backend/auth/password_reset.py

"""
Password reset functionality with email verification.

Flow:
1. User requests password reset
2. System generates secure token and sends email
3. User clicks link in email with token
4. User sets new password
5. Token is invalidated
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from backend.database import get_db
from backend.notifications.email_service import get_email_service
from backend.auth.auth_service import AuthService
from backend.monitoring.logger import log_info, log_warning
import hashlib


class PasswordResetService:
    """Service for handling password reset requests."""

    def __init__(self):
        self.token_expiry_hours = 24
        self._init_reset_table()

    def _init_reset_table(self):
        """Initialize password reset tokens table."""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                expires_at TIMESTAMP NOT NULL,
                used BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                used_at TIMESTAMP,
                ip_address TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reset_token ON password_reset_tokens(token_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reset_user ON password_reset_tokens(user_id)")

        conn.commit()
        conn.close()

    def _hash_token(self, token: str) -> str:
        """Hash reset token for storage."""
        return hashlib.sha256(token.encode()).hexdigest()

    def request_password_reset(self, email: str, ip_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Initiate password reset for user.

        Args:
            email: User's email address
            ip_address: IP address of requester

        Returns:
            Result dict with status
        """
        conn = get_db()
        cursor = conn.cursor()

        # Find user by email
        cursor.execute("SELECT id, username, email FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()

        if not user:
            conn.close()
            # Don't reveal if email exists or not (security)
            log_warning(f"[PASSWORD_RESET] Reset requested for non-existent email: {email}")
            return {
                'status': 'success',
                'message': 'If that email exists, a password reset link has been sent.'
            }

        user_id = user['id']
        username = user['username']

        # Generate secure token
        reset_token = secrets.token_urlsafe(32)
        token_hash = self._hash_token(reset_token)
        expires_at = datetime.now() + timedelta(hours=self.token_expiry_hours)

        # Save token
        cursor.execute("""
            INSERT INTO password_reset_tokens (user_id, token_hash, expires_at, ip_address)
            VALUES (?, ?, ?, ?)
        """, (user_id, token_hash, expires_at, ip_address))

        conn.commit()
        conn.close()

        # Send reset email
        email_service = get_email_service()
        reset_url = f"https://your-domain.com/reset-password?token={reset_token}"

        email_body = f"""
Hello {username},

You requested a password reset for your PineLab account.

Click the link below to reset your password:
{reset_url}

This link will expire in {self.token_expiry_hours} hours.

If you didn't request this reset, please ignore this email.

---
PineLab Security Team
        """.strip()

        try:
            email_service.send_email(
                to_email=email,
                subject="Password Reset Request - PineLab",
                body=email_body
            )
            log_info(f"[PASSWORD_RESET] Reset email sent to {email}")
        except Exception as e:
            log_warning(f"[PASSWORD_RESET] Failed to send email to {email}: {e}")
            # Don't fail the request if email fails

        return {
            'status': 'success',
            'message': 'If that email exists, a password reset link has been sent.',
            'expires_in_hours': self.token_expiry_hours
        }

    def validate_reset_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate password reset token.

        Args:
            token: Reset token from email link

        Returns:
            User info if valid, None otherwise
        """
        token_hash = self._hash_token(token)

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT rt.*, u.email, u.username
            FROM password_reset_tokens rt
            JOIN users u ON rt.user_id = u.id
            WHERE rt.token_hash = ?
        """, (token_hash,))

        record = cursor.fetchone()
        conn.close()

        if not record:
            log_warning("[PASSWORD_RESET] Invalid token attempted")
            return None

        # Check if already used
        if record['used']:
            log_warning(f"[PASSWORD_RESET] Already-used token attempted for user {record['user_id']}")
            return None

        # Check if expired
        expires_at = datetime.fromisoformat(record['expires_at'])
        if datetime.now() > expires_at:
            log_warning(f"[PASSWORD_RESET] Expired token attempted for user {record['user_id']}")
            return None

        return {
            'user_id': record['user_id'],
            'email': record['email'],
            'username': record['username']
        }

    def reset_password(self, token: str, new_password: str, ip_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Reset password using valid token.

        Args:
            token: Reset token from email
            new_password: New password to set
            ip_address: IP address of requester

        Returns:
            Result dict
        """
        # Validate token
        user_info = self.validate_reset_token(token)

        if not user_info:
            return {
                'status': 'error',
                'message': 'Invalid or expired reset token'
            }

        user_id = user_info['user_id']
        token_hash = self._hash_token(token)

        # Update password
        auth_service = AuthService()
        hashed_password = auth_service.hash_password(new_password)

        conn = get_db()
        cursor = conn.cursor()

        # Update user password
        cursor.execute("""
            UPDATE users
            SET hashed_password = ?
            WHERE id = ?
        """, (hashed_password, user_id))

        # Mark token as used
        cursor.execute("""
            UPDATE password_reset_tokens
            SET used = 1, used_at = ?
            WHERE token_hash = ?
        """, (datetime.now(), token_hash))

        conn.commit()
        conn.close()

        log_info(f"[PASSWORD_RESET] Password reset successful for user {user_id}")

        # Send confirmation email
        email_service = get_email_service()
        try:
            email_service.send_email(
                to_email=user_info['email'],
                subject="Password Changed - PineLab",
                body=f"""
Hello {user_info['username']},

Your password has been successfully changed.

If you didn't make this change, please contact support immediately.

---
PineLab Security Team
                """.strip()
            )
        except Exception as e:
            log_warning(f"[PASSWORD_RESET] Failed to send confirmation email: {e}")

        return {
            'status': 'success',
            'message': 'Password has been reset successfully'
        }

    def cleanup_expired_tokens(self):
        """Remove expired password reset tokens."""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM password_reset_tokens
            WHERE expires_at < ? OR used = 1
        """, (datetime.now() - timedelta(days=7),))  # Keep used tokens for 7 days for audit

        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        if deleted_count > 0:
            log_info(f"[PASSWORD_RESET] Cleaned up {deleted_count} expired reset tokens")

        return deleted_count


# Global password reset service
_password_reset_service: Optional[PasswordResetService] = None


def get_password_reset_service() -> PasswordResetService:
    """Get or create global password reset service instance."""
    global _password_reset_service
    if _password_reset_service is None:
        _password_reset_service = PasswordResetService()
    return _password_reset_service
