# backend/websocket/realtime_service.py

"""
WebSocket service for real-time updates.

Features:
- Real-time order status updates
- Live position changes
- Real-time P&L updates
- Trade execution notifications
- System alerts
- Authenticated connections
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set, Any, Optional, List
import json
import asyncio
from datetime import datetime
from backend.monitoring.logger import log_info, log_warning


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        # Active connections by user_id
        self.active_connections: Dict[int, Set[WebSocket]] = {}
        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, user_id: Optional[int] = None):
        """
        Accept WebSocket connection.

        Args:
            websocket: WebSocket connection
            user_id: User ID (None for anonymous)
        """
        await websocket.accept()

        # Store connection
        if user_id:
            if user_id not in self.active_connections:
                self.active_connections[user_id] = set()
            self.active_connections[user_id].add(websocket)

        # Store metadata
        self.connection_metadata[websocket] = {
            'user_id': user_id,
            'connected_at': datetime.now().isoformat(),
            'last_ping': datetime.now().isoformat()
        }

        log_info(f"[WEBSOCKET] Client connected" + (f" (user_id: {user_id})" if user_id else " (anonymous)"))

    def disconnect(self, websocket: WebSocket):
        """
        Remove WebSocket connection.

        Args:
            websocket: WebSocket connection
        """
        # Get user_id from metadata
        metadata = self.connection_metadata.get(websocket, {})
        user_id = metadata.get('user_id')

        # Remove from active connections
        if user_id and user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]

        # Remove metadata
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]

        log_info(f"[WEBSOCKET] Client disconnected" + (f" (user_id: {user_id})" if user_id else ""))

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """
        Send message to specific connection.

        Args:
            message: Message data
            websocket: Target websocket
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            log_warning(f"[WEBSOCKET] Failed to send message: {e}")

    async def send_to_user(self, message: Dict[str, Any], user_id: int):
        """
        Send message to all connections of a user.

        Args:
            message: Message data
            user_id: Target user ID
        """
        if user_id not in self.active_connections:
            return

        # Send to all user's connections
        dead_connections = set()
        for websocket in self.active_connections[user_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                log_warning(f"[WEBSOCKET] Failed to send to user {user_id}: {e}")
                dead_connections.add(websocket)

        # Clean up dead connections
        for websocket in dead_connections:
            self.disconnect(websocket)

    async def broadcast(self, message: Dict[str, Any], exclude_user_id: Optional[int] = None):
        """
        Broadcast message to all connected clients.

        Args:
            message: Message data
            exclude_user_id: Optional user ID to exclude from broadcast
        """
        dead_connections = set()

        for user_id, connections in list(self.active_connections.items()):
            if exclude_user_id and user_id == exclude_user_id:
                continue

            for websocket in connections:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    log_warning(f"[WEBSOCKET] Broadcast failed: {e}")
                    dead_connections.add(websocket)

        # Clean up dead connections
        for websocket in dead_connections:
            self.disconnect(websocket)

    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return sum(len(connections) for connections in self.active_connections.values())

    def get_user_connection_count(self, user_id: int) -> int:
        """Get number of connections for a user."""
        return len(self.active_connections.get(user_id, set()))

    async def ping_all(self):
        """Send ping to all connections to keep them alive."""
        message = {
            'type': 'ping',
            'timestamp': datetime.now().isoformat()
        }
        await self.broadcast(message)


# Global connection manager
_connection_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """Get or create global connection manager."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager


# Event types
class RealtimeEventType:
    """Event types for real-time updates."""
    ORDER_CREATED = "order_created"
    ORDER_UPDATED = "order_updated"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_UPDATED = "position_updated"
    PNL_UPDATED = "pnl_updated"
    TRADE_EXECUTED = "trade_executed"
    SYSTEM_ALERT = "system_alert"
    OPTIMIZATION_PROGRESS = "optimization_progress"
    AB_TEST_UPDATE = "ab_test_update"


# Event publishing functions
async def publish_order_event(event_type: str, order_data: Dict[str, Any], user_id: Optional[int] = None):
    """
    Publish order-related event.

    Args:
        event_type: Type of event
        order_data: Order data
        user_id: Target user ID (None = broadcast)
    """
    manager = get_connection_manager()

    message = {
        'type': event_type,
        'data': order_data,
        'timestamp': datetime.now().isoformat()
    }

    if user_id:
        await manager.send_to_user(message, user_id)
    else:
        await manager.broadcast(message)


async def publish_position_update(position_data: Dict[str, Any], user_id: Optional[int] = None):
    """Publish position update."""
    await publish_order_event(RealtimeEventType.POSITION_UPDATED, position_data, user_id)


async def publish_pnl_update(pnl_data: Dict[str, Any], user_id: Optional[int] = None):
    """Publish P&L update."""
    await publish_order_event(RealtimeEventType.PNL_UPDATED, pnl_data, user_id)


async def publish_system_alert(alert_message: str, severity: str = "info", user_id: Optional[int] = None):
    """
    Publish system alert.

    Args:
        alert_message: Alert message
        severity: Alert severity (info, warning, error, critical)
        user_id: Target user (None = broadcast to all)
    """
    manager = get_connection_manager()

    message = {
        'type': RealtimeEventType.SYSTEM_ALERT,
        'data': {
            'message': alert_message,
            'severity': severity
        },
        'timestamp': datetime.now().isoformat()
    }

    if user_id:
        await manager.send_to_user(message, user_id)
    else:
        await manager.broadcast(message)


async def publish_optimization_progress(
    strategy_name: str,
    trial_number: int,
    total_trials: int,
    best_value: float,
    user_id: int
):
    """Publish optimization progress update."""
    manager = get_connection_manager()

    message = {
        'type': RealtimeEventType.OPTIMIZATION_PROGRESS,
        'data': {
            'strategy_name': strategy_name,
            'trial_number': trial_number,
            'total_trials': total_trials,
            'best_value': best_value,
            'progress_percent': (trial_number / total_trials) * 100
        },
        'timestamp': datetime.now().isoformat()
    }

    await manager.send_to_user(message, user_id)


# Background task for keepalive pings
async def websocket_keepalive_task(interval_seconds: int = 30):
    """
    Background task to send keepalive pings.

    Args:
        interval_seconds: Ping interval in seconds
    """
    manager = get_connection_manager()

    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await manager.ping_all()
            log_info(f"[WEBSOCKET] Sent keepalive ping to {manager.get_connection_count()} connections")
        except Exception as e:
            log_warning(f"[WEBSOCKET] Keepalive error: {e}")


# WebSocket route handler helper
async def websocket_endpoint_handler(
    websocket: WebSocket,
    user_id: Optional[int] = None,
    on_message_callback: Optional[callable] = None
):
    """
    Generic WebSocket endpoint handler.

    Args:
        websocket: WebSocket connection
        user_id: Authenticated user ID (None for public)
        on_message_callback: Optional callback for incoming messages
    """
    manager = get_connection_manager()
    await manager.connect(websocket, user_id)

    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()

            # Parse message
            try:
                message = json.loads(data)

                # Handle message
                if on_message_callback:
                    await on_message_callback(websocket, message, user_id)
                else:
                    # Echo back by default
                    await manager.send_personal_message({
                        'type': 'echo',
                        'data': message
                    }, websocket)

            except json.JSONDecodeError:
                await manager.send_personal_message({
                    'type': 'error',
                    'message': 'Invalid JSON'
                }, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        log_warning(f"[WEBSOCKET] Connection error: {e}")
        manager.disconnect(websocket)
