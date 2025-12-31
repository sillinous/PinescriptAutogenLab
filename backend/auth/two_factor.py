# backend/auth/two_factor.py

"""
Two-Factor Authentication (2FA) using TOTP.

Features:
- TOTP-based 2FA (Google Authenticator, Authy, etc.)
- QR code generation for easy setup
- Backup codes for account recovery
- Optional per-user (can be enabled/disabled)
"""

import pyotp
import qrcode
import io
import base64
from datetime import datetime
from typing import Optional, Dict, Any, List
from backend.database import get_db
from backend.monitoring.logger import log_info, log_warning
from backend.monitoring.audit_log import get_audit_logger
import secrets


class TwoFactorAuthService:
    """Service for managing 2FA authentication."""

    def __init__(self):
        self.issuer_name = "PineLab"
        self._init_2fa_tables()

    def _init_2fa_tables(self):
        """Initialize 2FA tables."""
        conn = get_db()
        cursor = conn.cursor()

        # 2FA settings per user
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_2fa (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL UNIQUE,
                secret TEXT NOT NULL,
                enabled BOOLEAN DEFAULT 0,
                enabled_at TIMESTAMP,
                backup_codes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # 2FA verification attempts tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_2fa_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                success BOOLEAN NOT NULL,
                code_used TEXT,
                ip_address TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_2fa_user ON user_2fa(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_2fa_attempts_user ON user_2fa_attempts(user_id)")

        conn.commit()
        conn.close()

    def setup_2fa(self, user_id: int, username: str, email: str) -> Dict[str, Any]:
        """
        Initialize 2FA setup for user.

        Args:
            user_id: User ID
            username: Username for QR code
            email: User email for QR code

        Returns:
            Setup info including secret, QR code, and backup codes
        """
        conn = get_db()
        cursor = conn.cursor()

        # Check if 2FA already exists
        cursor.execute("SELECT * FROM user_2fa WHERE user_id = ?", (user_id,))
        existing = cursor.fetchone()

        if existing and existing['enabled']:
            conn.close()
            return {
                'status': 'error',
                'message': '2FA is already enabled. Disable it first to set up again.'
            }

        # Generate secret
        secret = pyotp.random_base32()

        # Generate backup codes (10 codes, 8 characters each)
        backup_codes = [secrets.token_hex(4).upper() for _ in range(10)]
        backup_codes_str = ','.join(backup_codes)

        # Save or update
        if existing:
            cursor.execute("""
                UPDATE user_2fa
                SET secret = ?, backup_codes = ?, enabled = 0
                WHERE user_id = ?
            """, (secret, backup_codes_str, user_id))
        else:
            cursor.execute("""
                INSERT INTO user_2fa (user_id, secret, backup_codes, enabled)
                VALUES (?, ?, ?, 0)
            """, (user_id, secret, backup_codes_str))

        conn.commit()
        conn.close()

        # Generate TOTP URI
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name=email,
            issuer_name=self.issuer_name
        )

        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)

        # Convert QR code to base64 image
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        qr_code_base64 = base64.b64encode(buffer.getvalue()).decode()

        log_info(f"[2FA] Setup initiated for user {user_id}")

        return {
            'status': 'success',
            'secret': secret,
            'qr_code': f"data:image/png;base64,{qr_code_base64}",
            'provisioning_uri': provisioning_uri,
            'backup_codes': backup_codes,
            'message': 'Scan QR code with authenticator app, then verify with a code to enable 2FA'
        }

    def verify_and_enable_2fa(
        self,
        user_id: int,
        verification_code: str,
        ip_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify TOTP code and enable 2FA.

        Args:
            user_id: User ID
            verification_code: 6-digit TOTP code from authenticator app
            ip_address: IP address of request

        Returns:
            Result dict
        """
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM user_2fa WHERE user_id = ?", (user_id,))
        record = cursor.fetchone()

        if not record:
            conn.close()
            return {
                'status': 'error',
                'message': '2FA not set up. Please set up 2FA first.'
            }

        if record['enabled']:
            conn.close()
            return {
                'status': 'error',
                'message': '2FA is already enabled'
            }

        # Verify code
        totp = pyotp.TOTP(record['secret'])
        is_valid = totp.verify(verification_code, valid_window=1)  # Allow 30s clock drift

        # Log attempt
        cursor.execute("""
            INSERT INTO user_2fa_attempts (user_id, success, code_used, ip_address)
            VALUES (?, ?, ?, ?)
        """, (user_id, is_valid, verification_code, ip_address))

        if not is_valid:
            conn.commit()
            conn.close()
            log_warning(f"[2FA] Failed verification attempt for user {user_id}")
            return {
                'status': 'error',
                'message': 'Invalid verification code'
            }

        # Enable 2FA
        cursor.execute("""
            UPDATE user_2fa
            SET enabled = 1, enabled_at = ?
            WHERE user_id = ?
        """, (datetime.now(), user_id))

        conn.commit()
        conn.close()

        # Audit log
        audit = get_audit_logger()
        audit.log_event(
            action_type='2fa_enabled',
            user_id=user_id,
            ip_address=ip_address
        )

        log_info(f"[2FA] Enabled successfully for user {user_id}")

        return {
            'status': 'success',
            'message': '2FA enabled successfully',
            'enabled': True
        }

    def verify_2fa_code(
        self,
        user_id: int,
        code: str,
        ip_address: Optional[str] = None,
        allow_backup_code: bool = True
    ) -> Dict[str, Any]:
        """
        Verify 2FA code during login.

        Args:
            user_id: User ID
            code: 6-digit TOTP code or backup code
            ip_address: IP address of request
            allow_backup_code: Whether to allow backup codes

        Returns:
            Verification result
        """
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM user_2fa WHERE user_id = ? AND enabled = 1", (user_id,))
        record = cursor.fetchone()

        if not record:
            conn.close()
            return {
                'status': 'error',
                'message': '2FA not enabled',
                'verified': False
            }

        # Try TOTP code first
        totp = pyotp.TOTP(record['secret'])
        is_valid = totp.verify(code, valid_window=1)

        used_backup = False

        # If TOTP fails, try backup codes
        if not is_valid and allow_backup_code:
            backup_codes = record['backup_codes'].split(',') if record['backup_codes'] else []
            if code.upper() in backup_codes:
                is_valid = True
                used_backup = True

                # Remove used backup code
                backup_codes.remove(code.upper())
                cursor.execute("""
                    UPDATE user_2fa
                    SET backup_codes = ?
                    WHERE user_id = ?
                """, (','.join(backup_codes), user_id))

                log_info(f"[2FA] Backup code used for user {user_id}. {len(backup_codes)} codes remaining.")

        # Log attempt
        cursor.execute("""
            INSERT INTO user_2fa_attempts (user_id, success, code_used, ip_address)
            VALUES (?, ?, ?, ?)
        """, (user_id, is_valid, code if not is_valid else '******', ip_address))

        conn.commit()
        conn.close()

        if is_valid:
            log_info(f"[2FA] Successful verification for user {user_id}" +
                    (" (backup code)" if used_backup else ""))
            return {
                'status': 'success',
                'verified': True,
                'used_backup_code': used_backup
            }
        else:
            log_warning(f"[2FA] Failed verification for user {user_id}")
            return {
                'status': 'error',
                'message': 'Invalid 2FA code',
                'verified': False
            }

    def disable_2fa(self, user_id: int, verification_code: str, ip_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Disable 2FA for user (requires verification).

        Args:
            user_id: User ID
            verification_code: Current TOTP code or backup code
            ip_address: IP address of request

        Returns:
            Result dict
        """
        # First verify the code
        verification = self.verify_2fa_code(user_id, verification_code, ip_address)

        if not verification.get('verified'):
            return {
                'status': 'error',
                'message': 'Invalid verification code. Cannot disable 2FA without valid code.'
            }

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE user_2fa
            SET enabled = 0
            WHERE user_id = ?
        """, (user_id,))

        conn.commit()
        conn.close()

        # Audit log
        audit = get_audit_logger()
        audit.log_event(
            action_type='2fa_disabled',
            user_id=user_id,
            ip_address=ip_address
        )

        log_info(f"[2FA] Disabled for user {user_id}")

        return {
            'status': 'success',
            'message': '2FA disabled successfully',
            'enabled': False
        }

    def regenerate_backup_codes(self, user_id: int, verification_code: str) -> Dict[str, Any]:
        """
        Regenerate backup codes (requires verification).

        Args:
            user_id: User ID
            verification_code: Current TOTP code

        Returns:
            New backup codes
        """
        # Verify code first
        verification = self.verify_2fa_code(user_id, verification_code, allow_backup_code=False)

        if not verification.get('verified'):
            return {
                'status': 'error',
                'message': 'Invalid verification code'
            }

        # Generate new backup codes
        backup_codes = [secrets.token_hex(4).upper() for _ in range(10)]
        backup_codes_str = ','.join(backup_codes)

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE user_2fa
            SET backup_codes = ?
            WHERE user_id = ?
        """, (backup_codes_str, user_id))

        conn.commit()
        conn.close()

        log_info(f"[2FA] Backup codes regenerated for user {user_id}")

        return {
            'status': 'success',
            'backup_codes': backup_codes,
            'message': 'New backup codes generated. Store them securely.'
        }

    def get_2fa_status(self, user_id: int) -> Dict[str, Any]:
        """
        Get 2FA status for user.

        Args:
            user_id: User ID

        Returns:
            2FA status info
        """
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM user_2fa WHERE user_id = ?", (user_id,))
        record = cursor.fetchone()

        if not record:
            conn.close()
            return {
                'enabled': False,
                'setup': False
            }

        backup_codes = record['backup_codes'].split(',') if record['backup_codes'] else []

        result = {
            'enabled': bool(record['enabled']),
            'setup': True,
            'enabled_at': record['enabled_at'],
            'backup_codes_remaining': len(backup_codes)
        }

        conn.close()
        return result

    def get_recent_attempts(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent 2FA verification attempts for user."""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM user_2fa_attempts
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, limit))

        attempts = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return attempts


# Global 2FA service
_two_factor_service: Optional[TwoFactorAuthService] = None


def get_two_factor_service() -> TwoFactorAuthService:
    """Get or create global 2FA service instance."""
    global _two_factor_service
    if _two_factor_service is None:
        _two_factor_service = TwoFactorAuthService()
    return _two_factor_service
