# backend/reliability/retry_handler.py

"""
Retry mechanism for failed operations with exponential backoff.

Handles:
- Failed webhook deliveries
- Failed order submissions
- Failed API calls to brokers
- Failed email notifications
"""

import time
import asyncio
from typing import Callable, Any, Optional, Dict
from datetime import datetime, timedelta
from backend.database import get_db
from backend.monitoring.logger import log_error, log_info
import traceback


class RetryHandler:
    """
    Exponential backoff retry handler.

    Features:
    - Configurable max attempts
    - Exponential backoff with jitter
    - Dead letter queue for permanently failed items
    - Retry history tracking
    """

    def __init__(
        self,
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 300.0,
        exponential_base: float = 2.0
    ):
        """
        Initialize retry handler.

        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay between retries
            exponential_base: Base for exponential backoff (default 2.0)
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self._init_retry_tables()

    def _init_retry_tables(self):
        """Initialize database tables for retry tracking."""
        conn = get_db()
        cursor = conn.cursor()

        # Retry queue table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS retry_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_type TEXT NOT NULL,
                operation_data TEXT NOT NULL,
                attempt_count INTEGER DEFAULT 0,
                max_attempts INTEGER DEFAULT 5,
                last_attempt_at TIMESTAMP,
                last_error TEXT,
                next_retry_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending'
            )
        """)

        # Dead letter queue for permanent failures
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dead_letter_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_type TEXT NOT NULL,
                operation_data TEXT NOT NULL,
                total_attempts INTEGER,
                final_error TEXT,
                failed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                original_created_at TIMESTAMP
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_retry_next_retry ON retry_queue(next_retry_at, status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_retry_type ON retry_queue(operation_type)")

        conn.commit()
        conn.close()

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt using exponential backoff.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )

        # Add jitter (±20%)
        import random
        jitter = delay * 0.2 * (random.random() * 2 - 1)
        return max(0.1, delay + jitter)

    async def retry_async(
        self,
        func: Callable,
        *args,
        operation_type: str = "unknown",
        **kwargs
    ) -> Any:
        """
        Retry an async function with exponential backoff.

        Args:
            func: Async function to retry
            *args: Positional arguments for func
            operation_type: Type of operation (for logging)
            **kwargs: Keyword arguments for func

        Returns:
            Result of successful function call

        Raises:
            Exception: If all retries fail
        """
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                result = await func(*args, **kwargs)
                if attempt > 0:
                    log_info(f"[RETRY] {operation_type} succeeded on attempt {attempt + 1}")
                return result

            except Exception as e:
                last_exception = e
                if attempt < self.max_attempts - 1:
                    delay = self.calculate_delay(attempt)
                    log_error(
                        f"[RETRY] {operation_type} failed (attempt {attempt + 1}/{self.max_attempts}). "
                        f"Retrying in {delay:.1f}s. Error: {str(e)}"
                    )
                    await asyncio.sleep(delay)
                else:
                    log_error(
                        f"[RETRY] {operation_type} permanently failed after {self.max_attempts} attempts. "
                        f"Final error: {str(e)}"
                    )

        raise last_exception

    def retry_sync(
        self,
        func: Callable,
        *args,
        operation_type: str = "unknown",
        **kwargs
    ) -> Any:
        """
        Retry a synchronous function with exponential backoff.

        Args:
            func: Function to retry
            *args: Positional arguments for func
            operation_type: Type of operation (for logging)
            **kwargs: Keyword arguments for func

        Returns:
            Result of successful function call

        Raises:
            Exception: If all retries fail
        """
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    log_info(f"[RETRY] {operation_type} succeeded on attempt {attempt + 1}")
                return result

            except Exception as e:
                last_exception = e
                if attempt < self.max_attempts - 1:
                    delay = self.calculate_delay(attempt)
                    log_error(
                        f"[RETRY] {operation_type} failed (attempt {attempt + 1}/{self.max_attempts}). "
                        f"Retrying in {delay:.1f}s. Error: {str(e)}"
                    )
                    time.sleep(delay)
                else:
                    log_error(
                        f"[RETRY] {operation_type} permanently failed after {self.max_attempts} attempts. "
                        f"Final error: {str(e)}"
                    )

        raise last_exception

    def queue_for_retry(
        self,
        operation_type: str,
        operation_data: Dict[str, Any],
        max_attempts: Optional[int] = None
    ) -> int:
        """
        Add operation to retry queue for background processing.

        Args:
            operation_type: Type of operation (e.g., 'webhook', 'order', 'email')
            operation_data: Serializable data needed to retry operation
            max_attempts: Override default max attempts

        Returns:
            Queue entry ID
        """
        import json

        conn = get_db()
        cursor = conn.cursor()

        next_retry = datetime.now() + timedelta(seconds=self.base_delay)

        cursor.execute("""
            INSERT INTO retry_queue (operation_type, operation_data, max_attempts, next_retry_at, status)
            VALUES (?, ?, ?, ?, 'pending')
        """, (
            operation_type,
            json.dumps(operation_data),
            max_attempts or self.max_attempts,
            next_retry
        ))

        queue_id = cursor.lastrowid
        conn.commit()
        conn.close()

        log_info(f"[RETRY] Queued {operation_type} for retry (ID: {queue_id})")
        return queue_id

    def move_to_dead_letter_queue(self, retry_id: int):
        """
        Move permanently failed item to dead letter queue.

        Args:
            retry_id: Retry queue entry ID
        """
        conn = get_db()
        cursor = conn.cursor()

        # Get retry entry
        cursor.execute("SELECT * FROM retry_queue WHERE id = ?", (retry_id,))
        entry = cursor.fetchone()

        if not entry:
            conn.close()
            return

        # Move to dead letter queue
        cursor.execute("""
            INSERT INTO dead_letter_queue (operation_type, operation_data, total_attempts, final_error, original_created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            entry['operation_type'],
            entry['operation_data'],
            entry['attempt_count'],
            entry['last_error'],
            entry['created_at']
        ))

        # Delete from retry queue
        cursor.execute("DELETE FROM retry_queue WHERE id = ?", (retry_id,))

        conn.commit()
        conn.close()

        log_error(f"[RETRY] Moved operation {retry_id} to dead letter queue after {entry['attempt_count']} failed attempts")

    def get_pending_retries(self, limit: int = 100) -> list:
        """
        Get pending retry operations ready to be retried.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of retry entries ready for processing
        """
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM retry_queue
            WHERE status = 'pending'
            AND (next_retry_at IS NULL OR next_retry_at <= ?)
            ORDER BY next_retry_at ASC
            LIMIT ?
        """, (datetime.now(), limit))

        entries = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return entries


# Global retry handler instance
_retry_handler: Optional[RetryHandler] = None


def get_retry_handler() -> RetryHandler:
    """Get or create global retry handler instance."""
    global _retry_handler
    if _retry_handler is None:
        _retry_handler = RetryHandler()
    return _retry_handler


# Convenience decorators for retrying functions
def retry_async_operation(operation_type: str = "async_operation", max_attempts: int = 5):
    """Decorator for async functions with automatic retry."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            handler = get_retry_handler()
            handler.max_attempts = max_attempts
            return await handler.retry_async(func, *args, operation_type=operation_type, **kwargs)
        return wrapper
    return decorator


def retry_sync_operation(operation_type: str = "sync_operation", max_attempts: int = 5):
    """Decorator for sync functions with automatic retry."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            handler = get_retry_handler()
            handler.max_attempts = max_attempts
            return handler.retry_sync(func, *args, operation_type=operation_type, **kwargs)
        return wrapper
    return decorator
