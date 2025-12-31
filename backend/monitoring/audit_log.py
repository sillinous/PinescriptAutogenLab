# backend/monitoring/audit_log.py

"""
Audit logging system for tracking sensitive actions.

Logs:
- User authentication events
- Admin actions
- Credential changes
- Configuration changes
- Order modifications
- Subscription changes
"""

from datetime import datetime
from typing import Optional, Dict, Any
from backend.database import get_db
from backend.monitoring.logger import log_info
import json


class AuditLogger:
    """Service for logging auditable events."""

    def __init__(self):
        self._init_audit_table()

    def _init_audit_table(self):
        """Initialize audit log table."""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                username TEXT,
                action_type TEXT NOT NULL,
                resource_type TEXT,
                resource_id TEXT,
                action_details TEXT,
                ip_address TEXT,
                user_agent TEXT,
                status TEXT DEFAULT 'success',
                error_message TEXT
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_log(resource_type, resource_id)")

        conn.commit()
        conn.close()

    def log_event(
        self,
        action_type: str,
        user_id: Optional[int] = None,
        username: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action_details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        status: str = "success",
        error_message: Optional[str] = None
    ) -> int:
        """
        Log an auditable event.

        Args:
            action_type: Type of action (e.g., 'login', 'user_created', 'order_modified')
            user_id: ID of user performing action
            username: Username performing action
            resource_type: Type of resource affected (e.g., 'user', 'order', 'credential')
            resource_id: ID of resource affected
            action_details: Additional details (will be JSON-serialized)
            ip_address: IP address of requester
            user_agent: User agent string
            status: 'success' or 'failure'
            error_message: Error message if failed

        Returns:
            Audit log entry ID
        """
        conn = get_db()
        cursor = conn.cursor()

        details_json = json.dumps(action_details) if action_details else None

        cursor.execute("""
            INSERT INTO audit_log (
                user_id, username, action_type, resource_type, resource_id,
                action_details, ip_address, user_agent, status, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, username, action_type, resource_type, resource_id,
            details_json, ip_address, user_agent, status, error_message
        ))

        log_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Also log to console for critical actions
        if action_type in ['login', 'failed_login', 'user_created', 'user_deleted', 'admin_action']:
            log_info(
                f"[AUDIT] {action_type.upper()}: user={username or user_id}, "
                f"resource={resource_type}:{resource_id}, status={status}"
            )

        return log_id

    # Convenience methods for common audit events

    def log_login(self, user_id: int, username: str, ip_address: str, user_agent: str, success: bool = True):
        """Log user login attempt."""
        return self.log_event(
            action_type='login' if success else 'failed_login',
            user_id=user_id if success else None,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            status='success' if success else 'failure'
        )

    def log_logout(self, user_id: int, username: str, ip_address: str):
        """Log user logout."""
        return self.log_event(
            action_type='logout',
            user_id=user_id,
            username=username,
            ip_address=ip_address
        )

    def log_user_created(self, admin_user_id: int, admin_username: str, new_user_id: int, new_username: str, ip_address: str):
        """Log user creation."""
        return self.log_event(
            action_type='user_created',
            user_id=admin_user_id,
            username=admin_username,
            resource_type='user',
            resource_id=str(new_user_id),
            action_details={'new_username': new_username},
            ip_address=ip_address
        )

    def log_user_modified(self, admin_user_id: int, admin_username: str, target_user_id: int, changes: Dict[str, Any], ip_address: str):
        """Log user modification."""
        return self.log_event(
            action_type='user_modified',
            user_id=admin_user_id,
            username=admin_username,
            resource_type='user',
            resource_id=str(target_user_id),
            action_details=changes,
            ip_address=ip_address
        )

    def log_user_deleted(self, admin_user_id: int, admin_username: str, deleted_user_id: int, ip_address: str):
        """Log user deletion."""
        return self.log_event(
            action_type='user_deleted',
            user_id=admin_user_id,
            username=admin_username,
            resource_type='user',
            resource_id=str(deleted_user_id),
            ip_address=ip_address
        )

    def log_password_change(self, user_id: int, username: str, ip_address: str):
        """Log password change."""
        return self.log_event(
            action_type='password_changed',
            user_id=user_id,
            username=username,
            resource_type='user',
            resource_id=str(user_id),
            ip_address=ip_address
        )

    def log_credential_added(self, user_id: int, username: str, broker_type: str, ip_address: str):
        """Log broker credential addition."""
        return self.log_event(
            action_type='credential_added',
            user_id=user_id,
            username=username,
            resource_type='credential',
            resource_id=broker_type,
            action_details={'broker_type': broker_type},
            ip_address=ip_address
        )

    def log_order_created(self, user_id: Optional[int], order_id: int, symbol: str, side: str, ip_address: str):
        """Log order creation."""
        return self.log_event(
            action_type='order_created',
            user_id=user_id,
            resource_type='order',
            resource_id=str(order_id),
            action_details={'symbol': symbol, 'side': side},
            ip_address=ip_address
        )

    def log_admin_action(self, admin_user_id: int, admin_username: str, action: str, details: Dict[str, Any], ip_address: str):
        """Log general admin action."""
        return self.log_event(
            action_type='admin_action',
            user_id=admin_user_id,
            username=admin_username,
            action_details={'action': action, **details},
            ip_address=ip_address
        )

    def get_user_audit_log(self, user_id: int, limit: int = 100) -> list:
        """Get audit log entries for a specific user."""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM audit_log
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, limit))

        entries = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return entries

    def get_recent_audit_log(self, limit: int = 100, action_type: Optional[str] = None) -> list:
        """Get recent audit log entries."""
        conn = get_db()
        cursor = conn.cursor()

        if action_type:
            cursor.execute("""
                SELECT * FROM audit_log
                WHERE action_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (action_type, limit))
        else:
            cursor.execute("""
                SELECT * FROM audit_log
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

        entries = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return entries

    def get_failed_login_attempts(self, username: str, hours: int = 24) -> int:
        """Get count of failed login attempts for user in last N hours."""
        conn = get_db()
        cursor = conn.cursor()

        cutoff = datetime.now() - timedelta(hours=hours)

        cursor.execute("""
            SELECT COUNT(*) as count FROM audit_log
            WHERE action_type = 'failed_login'
            AND username = ?
            AND timestamp > ?
        """, (username, cutoff))

        result = cursor.fetchone()
        conn.close()

        return result['count'] if result else 0


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get or create global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


# Convenience function
def log_audit_event(*args, **kwargs):
    """Convenience function to log audit event."""
    logger = get_audit_logger()
    return logger.log_event(*args, **kwargs)
