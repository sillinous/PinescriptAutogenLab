# backend/auth/email_verification.py

"""
Email verification system for new user registrations.

Flow:
1. User registers with email
2. System sends verification email with token
3. User clicks verification link
4. Account is activated
5. User can login
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from backend.database import get_db
from backend.notifications.email_service import get_email_service
from backend.monitoring.logger import log_info, log_warning, log_error
import hashlib


class EmailVerificationService:
    """Service for email verification of new users."""

    def __init__(self):
        self.token_expiry_hours = 48  # 48 hours to verify email
        self._init_verification_table()

    def _init_verification_table(self):
        """Initialize email verification tokens table."""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS email_verification_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                verified BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                verified_at TIMESTAMP,
                ip_address TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_verify_token ON email_verification_tokens(token_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_verify_user ON email_verification_tokens(user_id)")

        conn.commit()
        conn.close()

    def _hash_token(self, token: str) -> str:
        """Hash verification token for storage."""
        return hashlib.sha256(token.encode()).hexdigest()

    def send_verification_email(
        self,
        user_id: int,
        email: str,
        username: str,
        ip_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send verification email to newly registered user.

        Args:
            user_id: User ID
            email: User's email address
            username: Username for personalization
            ip_address: IP address of registration

        Returns:
            Result dict with status
        """
        # Generate secure token
        verification_token = secrets.token_urlsafe(32)
        token_hash = self._hash_token(verification_token)
        expires_at = datetime.now() + timedelta(hours=self.token_expiry_hours)

        # Save token
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO email_verification_tokens (user_id, token_hash, email, expires_at, ip_address)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, token_hash, email, expires_at, ip_address))

        conn.commit()
        conn.close()

        # Send verification email
        email_service = get_email_service()
        verification_url = f"https://your-domain.com/verify-email?token={verification_token}"

        email_body = f"""
Welcome to PineLab, {username}!

Thank you for registering. Please verify your email address by clicking the link below:

{verification_url}

This link will expire in {self.token_expiry_hours} hours.

If you didn't create this account, please ignore this email.

---
PineLab Team
        """.strip()

        try:
            email_service.send_email(
                to_email=email,
                subject="Verify Your Email - PineLab",
                body=email_body
            )
            log_info(f"[EMAIL_VERIFICATION] Verification email sent to {email} (user_id: {user_id})")

            return {
                'status': 'success',
                'message': 'Verification email sent',
                'expires_in_hours': self.token_expiry_hours
            }
        except Exception as e:
            log_error(f"[EMAIL_VERIFICATION] Failed to send email to {email}: {e}")
            return {
                'status': 'error',
                'message': 'Failed to send verification email',
                'error': str(e)
            }

    def verify_email(self, token: str, ip_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify user's email using token.

        Args:
            token: Verification token from email link
            ip_address: IP address of verification request

        Returns:
            Result dict with status
        """
        token_hash = self._hash_token(token)

        conn = get_db()
        cursor = conn.cursor()

        # Get verification record
        cursor.execute("""
            SELECT * FROM email_verification_tokens
            WHERE token_hash = ?
        """, (token_hash,))

        record = cursor.fetchone()

        if not record:
            conn.close()
            log_warning("[EMAIL_VERIFICATION] Invalid verification token attempted")
            return {
                'status': 'error',
                'message': 'Invalid verification token'
            }

        # Check if already verified
        if record['verified']:
            conn.close()
            log_warning(f"[EMAIL_VERIFICATION] Already-verified token attempted for user {record['user_id']}")
            return {
                'status': 'error',
                'message': 'Email already verified'
            }

        # Check if expired
        expires_at = datetime.fromisoformat(record['expires_at'])
        if datetime.now() > expires_at:
            conn.close()
            log_warning(f"[EMAIL_VERIFICATION] Expired token attempted for user {record['user_id']}")
            return {
                'status': 'error',
                'message': 'Verification token has expired. Please request a new one.'
            }

        user_id = record['user_id']
        email = record['email']

        # Mark token as verified
        cursor.execute("""
            UPDATE email_verification_tokens
            SET verified = 1, verified_at = ?
            WHERE token_hash = ?
        """, (datetime.now(), token_hash))

        # Mark user as verified
        cursor.execute("""
            UPDATE users
            SET is_verified = 1
            WHERE id = ?
        """, (user_id,))

        conn.commit()
        conn.close()

        log_info(f"[EMAIL_VERIFICATION] Email verified successfully for user {user_id} ({email})")

        # Send welcome email
        email_service = get_email_service()
        try:
            cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
            user = cursor.fetchone()
            username = user['username'] if user else 'User'

            email_service.send_email(
                to_email=email,
                subject="Welcome to PineLab!",
                body=f"""
Hello {username},

Your email has been verified successfully! Welcome to PineLab.

You can now:
- Connect your broker accounts
- Set up automated trading strategies
- Run optimizations and A/B tests
- Track your trading performance

Get started: https://your-domain.com/dashboard

Happy trading!

---
PineLab Team
                """.strip()
            )
        except Exception as e:
            log_warning(f"[EMAIL_VERIFICATION] Failed to send welcome email: {e}")

        return {
            'status': 'success',
            'message': 'Email verified successfully',
            'user_id': user_id
        }

    def resend_verification_email(
        self,
        user_id: int,
        ip_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Resend verification email to user.

        Args:
            user_id: User ID
            ip_address: IP address of request

        Returns:
            Result dict
        """
        conn = get_db()
        cursor = conn.cursor()

        # Get user details
        cursor.execute("SELECT email, username, is_verified FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()

        if not user:
            conn.close()
            return {
                'status': 'error',
                'message': 'User not found'
            }

        if user['is_verified']:
            conn.close()
            return {
                'status': 'error',
                'message': 'Email already verified'
            }

        email = user['email']
        username = user['username']

        # Invalidate old tokens
        cursor.execute("""
            UPDATE email_verification_tokens
            SET verified = 1
            WHERE user_id = ? AND verified = 0
        """, (user_id,))

        conn.commit()
        conn.close()

        # Send new verification email
        return self.send_verification_email(user_id, email, username, ip_address)

    def check_verification_status(self, user_id: int) -> Dict[str, Any]:
        """
        Check email verification status for user.

        Args:
            user_id: User ID

        Returns:
            Verification status info
        """
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("SELECT is_verified, email FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()

        if not user:
            conn.close()
            return {
                'status': 'error',
                'message': 'User not found'
            }

        is_verified = bool(user['is_verified'])

        # Get pending verification token info
        cursor.execute("""
            SELECT expires_at FROM email_verification_tokens
            WHERE user_id = ? AND verified = 0
            ORDER BY created_at DESC
            LIMIT 1
        """, (user_id,))

        pending_token = cursor.fetchone()
        conn.close()

        result = {
            'user_id': user_id,
            'email': user['email'],
            'is_verified': is_verified
        }

        if pending_token and not is_verified:
            expires_at = datetime.fromisoformat(pending_token['expires_at'])
            result['pending_verification'] = True
            result['token_expires_at'] = pending_token['expires_at']
            result['token_expired'] = datetime.now() > expires_at
        else:
            result['pending_verification'] = False

        return result

    def cleanup_expired_tokens(self):
        """Remove expired verification tokens."""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM email_verification_tokens
            WHERE expires_at < ? OR verified = 1
        """, (datetime.now() - timedelta(days=7),))  # Keep verified tokens for 7 days for audit

        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        if deleted_count > 0:
            log_info(f"[EMAIL_VERIFICATION] Cleaned up {deleted_count} expired verification tokens")

        return deleted_count

    def require_verification_middleware(self, user: Dict[str, Any]) -> bool:
        """
        Check if user needs email verification.

        Can be used as middleware to block unverified users.

        Args:
            user: User dict from authentication

        Returns:
            True if verified, raises exception if not
        """
        if not user.get('is_verified'):
            from backend.middleware import AuthorizationError
            raise AuthorizationError(
                "Email verification required. Please check your email and verify your account."
            )
        return True


# Global email verification service
_email_verification_service: Optional[EmailVerificationService] = None


def get_email_verification_service() -> EmailVerificationService:
    """Get or create global email verification service instance."""
    global _email_verification_service
    if _email_verification_service is None:
        _email_verification_service = EmailVerificationService()
    return _email_verification_service
