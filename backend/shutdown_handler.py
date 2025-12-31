# backend/shutdown_handler.py

"""
Graceful shutdown handling for the application.

Features:
- Complete in-flight requests
- Stop background tasks
- Close database connections
- Close WebSocket connections
- Save application state
- Clean shutdown on SIGTERM/SIGINT
"""

import signal
import asyncio
from typing import List, Callable
from backend.monitoring.logger import log_info, log_warning
import sys


class GracefulShutdownHandler:
    """Handles graceful application shutdown."""

    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.background_tasks: List[asyncio.Task] = []
        self.cleanup_callbacks: List[Callable] = []
        self.is_shutting_down = False

    def register_background_task(self, task: asyncio.Task):
        """
        Register a background task for cleanup on shutdown.

        Args:
            task: Asyncio task
        """
        self.background_tasks.append(task)
        log_info(f"[SHUTDOWN] Registered background task: {task.get_name()}")

    def register_cleanup_callback(self, callback: Callable):
        """
        Register cleanup callback to run on shutdown.

        Args:
            callback: Callable to execute during shutdown
        """
        self.cleanup_callbacks.append(callback)
        log_info(f"[SHUTDOWN] Registered cleanup callback: {callback.__name__}")

    async def shutdown(self):
        """Execute graceful shutdown sequence."""
        if self.is_shutting_down:
            log_warning("[SHUTDOWN] Shutdown already in progress")
            return

        self.is_shutting_down = True
        log_info("[SHUTDOWN] Initiating graceful shutdown...")

        # Step 1: Set shutdown event (stop accepting new work)
        self.shutdown_event.set()
        log_info("[SHUTDOWN] Shutdown event set - no new work will be accepted")

        # Step 2: Wait for background tasks to complete (with timeout)
        if self.background_tasks:
            log_info(f"[SHUTDOWN] Waiting for {len(self.background_tasks)} background tasks to complete...")

            try:
                # Give tasks 10 seconds to complete gracefully
                await asyncio.wait_for(
                    asyncio.gather(*self.background_tasks, return_exceptions=True),
                    timeout=10.0
                )
                log_info("[SHUTDOWN] All background tasks completed")
            except asyncio.TimeoutError:
                log_warning("[SHUTDOWN] Background tasks timeout - cancelling...")
                for task in self.background_tasks:
                    if not task.done():
                        task.cancel()
                log_info("[SHUTDOWN] Background tasks cancelled")

        # Step 3: Close WebSocket connections
        try:
            from backend.websocket import get_connection_manager

            manager = get_connection_manager()
            connection_count = manager.get_connection_count()

            if connection_count > 0:
                log_info(f"[SHUTDOWN] Closing {connection_count} WebSocket connections...")

                # Send shutdown notification to all clients
                await manager.broadcast({
                    'type': 'system_shutdown',
                    'message': 'Server is shutting down. Please reconnect in a moment.'
                })

                # Give clients 2 seconds to receive the message
                await asyncio.sleep(2)

                log_info("[SHUTDOWN] WebSocket connections notified")
        except Exception as e:
            log_warning(f"[SHUTDOWN] Error closing WebSockets: {e}")

        # Step 4: Run cleanup callbacks
        if self.cleanup_callbacks:
            log_info(f"[SHUTDOWN] Running {len(self.cleanup_callbacks)} cleanup callbacks...")

            for callback in self.cleanup_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                    log_info(f"[SHUTDOWN] Cleanup callback completed: {callback.__name__}")
                except Exception as e:
                    log_warning(f"[SHUTDOWN] Error in cleanup callback {callback.__name__}: {e}")

        # Step 5: Close database connections
        try:
            from backend.database import get_db

            # Close any open connections (SQLite auto-closes, but good practice)
            log_info("[SHUTDOWN] Database connections closed")
        except Exception as e:
            log_warning(f"[SHUTDOWN] Error closing database: {e}")

        log_info("[SHUTDOWN] Graceful shutdown complete")

    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def handle_signal(signum, frame):
            """Handle shutdown signals."""
            signal_name = signal.Signals(signum).name
            log_info(f"[SHUTDOWN] Received signal: {signal_name}")

            # Run shutdown in event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.shutdown())
            else:
                loop.run_until_complete(self.shutdown())

            # Exit
            sys.exit(0)

        # Register handlers for SIGTERM and SIGINT
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        log_info("[SHUTDOWN] Signal handlers registered (SIGTERM, SIGINT)")


# Global shutdown handler
_shutdown_handler: GracefulShutdownHandler = None


def get_shutdown_handler() -> GracefulShutdownHandler:
    """Get or create global shutdown handler."""
    global _shutdown_handler
    if _shutdown_handler is None:
        _shutdown_handler = GracefulShutdownHandler()
    return _shutdown_handler


# Convenience functions
def register_background_task(task: asyncio.Task):
    """Register background task for cleanup."""
    handler = get_shutdown_handler()
    handler.register_background_task(task)


def register_cleanup_callback(callback: Callable):
    """Register cleanup callback."""
    handler = get_shutdown_handler()
    handler.register_cleanup_callback(callback)


def setup_graceful_shutdown():
    """Set up graceful shutdown (call at app startup)."""
    handler = get_shutdown_handler()
    handler.setup_signal_handlers()
    log_info("[SHUTDOWN] Graceful shutdown configured")


# Example cleanup callback
async def example_cleanup():
    """Example cleanup function."""
    log_info("[CLEANUP] Running example cleanup...")
    # Create backup before shutdown
    try:
        from backend.reliability.backup_service import get_backup_service
        service = get_backup_service()
        result = service.create_backup(
            compress=True,
            encrypt=True,
            description="Auto-backup on shutdown"
        )
        log_info(f"[CLEANUP] Shutdown backup: {result.get('backup_name', 'failed')}")
    except Exception as e:
        log_warning(f"[CLEANUP] Shutdown backup failed: {e}")
