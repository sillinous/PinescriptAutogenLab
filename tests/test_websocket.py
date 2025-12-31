# tests/test_websocket.py

"""
Tests for WebSocket real-time updates.
"""

import pytest
import asyncio
import json
from fastapi.testclient import TestClient


@pytest.mark.websocket
class TestWebSocketConnections:
    """Test WebSocket connection management."""

    def test_websocket_connection(self, client):
        """Test basic WebSocket connection."""
        with client.websocket_connect("/ws") as websocket:
            # Should connect successfully
            assert websocket is not None

            # Send ping
            websocket.send_json({'type': 'ping'})

            # Should receive response
            data = websocket.receive_json(timeout=5)
            assert data is not None

    def test_websocket_echo(self, client):
        """Test WebSocket echo functionality."""
        with client.websocket_connect("/ws") as websocket:
            test_message = {'type': 'test', 'data': 'hello'}

            websocket.send_json(test_message)
            response = websocket.receive_json(timeout=5)

            assert response is not None
            assert 'type' in response

    def test_websocket_user_connection(self, client, user_token):
        """Test user-specific WebSocket connection."""
        # WebSocket with user ID would require authentication
        # This test shows the pattern
        # with client.websocket_connect(f"/ws/1?token={user_token}") as websocket:
        #     assert websocket is not None

    def test_multiple_websocket_connections(self, client):
        """Test multiple simultaneous WebSocket connections."""
        with client.websocket_connect("/ws") as ws1, \
             client.websocket_connect("/ws") as ws2:

            # Both should be connected
            assert ws1 is not None
            assert ws2 is not None

            # Send to first
            ws1.send_json({'type': 'test', 'message': 'from ws1'})

            # Receive on first
            data1 = ws1.receive_json(timeout=5)
            assert data1 is not None

    def test_websocket_disconnect(self, client):
        """Test WebSocket disconnect handling."""
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({'type': 'ping'})
            data = websocket.receive_json(timeout=5)
            assert data is not None

        # Connection should be closed after exiting context

    def test_websocket_invalid_json(self, client):
        """Test WebSocket handling of invalid JSON."""
        with client.websocket_connect("/ws") as websocket:
            # Send invalid JSON
            websocket.send_text("invalid json {{{")

            # Should receive error message
            response = websocket.receive_json(timeout=5)
            assert response is not None
            # Should indicate error
            assert 'error' in response.get('type', '') or 'error' in str(response)


@pytest.mark.websocket
class TestWebSocketRealtimeUpdates:
    """Test real-time updates via WebSocket."""

    @pytest.mark.asyncio
    async def test_order_update_broadcast(self, db):
        """Test broadcasting order updates."""
        from backend.websocket import publish_order_event, RealtimeEventType

        # This would require actual WebSocket connections to test properly
        # Here we just test the publish function doesn't error
        await publish_order_event(
            event_type=RealtimeEventType.ORDER_CREATED,
            order_data={
                'order_id': 1,
                'symbol': 'AAPL',
                'action': 'buy',
                'quantity': 10,
                'status': 'pending'
            }
        )

    @pytest.mark.asyncio
    async def test_position_update_broadcast(self, db):
        """Test broadcasting position updates."""
        from backend.websocket import publish_position_update

        await publish_position_update(
            position_data={
                'symbol': 'AAPL',
                'quantity': 100,
                'avg_price': 150.0,
                'current_price': 155.0,
                'unrealized_pnl': 500.0
            }
        )

    @pytest.mark.asyncio
    async def test_pnl_update_broadcast(self, db):
        """Test broadcasting P&L updates."""
        from backend.websocket import publish_pnl_update

        await publish_pnl_update(
            pnl_data={
                'total_pnl': 1500.0,
                'realized_pnl': 1000.0,
                'unrealized_pnl': 500.0,
                'win_rate': 0.65
            }
        )

    @pytest.mark.asyncio
    async def test_system_alert_broadcast(self, db):
        """Test broadcasting system alerts."""
        from backend.websocket import publish_system_alert

        await publish_system_alert(
            alert_message="Test system alert",
            severity="warning"
        )

    @pytest.mark.asyncio
    async def test_optimization_progress_broadcast(self, db):
        """Test broadcasting optimization progress."""
        from backend.websocket import publish_optimization_progress

        await publish_optimization_progress(
            strategy_name="TestStrategy",
            trial_number=50,
            total_trials=100,
            best_value=0.85,
            user_id=1
        )


@pytest.mark.websocket
class TestConnectionManager:
    """Test WebSocket ConnectionManager."""

    @pytest.mark.asyncio
    async def test_connection_manager_connect(self):
        """Test connection manager connect."""
        from backend.websocket import ConnectionManager
        from unittest.mock import Mock, AsyncMock

        manager = ConnectionManager()

        # Mock WebSocket
        mock_ws = Mock()
        mock_ws.accept = AsyncMock()

        await manager.connect(mock_ws, user_id=1)

        assert 1 in manager.active_connections
        assert mock_ws in manager.active_connections[1]

    @pytest.mark.asyncio
    async def test_connection_manager_disconnect(self):
        """Test connection manager disconnect."""
        from backend.websocket import ConnectionManager
        from unittest.mock import Mock, AsyncMock

        manager = ConnectionManager()

        mock_ws = Mock()
        mock_ws.accept = AsyncMock()

        await manager.connect(mock_ws, user_id=1)
        manager.disconnect(mock_ws)

        assert 1 not in manager.active_connections or \
               mock_ws not in manager.active_connections.get(1, set())

    @pytest.mark.asyncio
    async def test_connection_manager_send_to_user(self):
        """Test sending message to specific user."""
        from backend.websocket import ConnectionManager
        from unittest.mock import Mock, AsyncMock

        manager = ConnectionManager()

        mock_ws = Mock()
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()

        await manager.connect(mock_ws, user_id=1)

        message = {'type': 'test', 'data': 'hello'}
        await manager.send_to_user(message, user_id=1)

        mock_ws.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_connection_manager_broadcast(self):
        """Test broadcasting to all connections."""
        from backend.websocket import ConnectionManager
        from unittest.mock import Mock, AsyncMock

        manager = ConnectionManager()

        # Create multiple mock connections
        mock_ws1 = Mock()
        mock_ws1.accept = AsyncMock()
        mock_ws1.send_json = AsyncMock()

        mock_ws2 = Mock()
        mock_ws2.accept = AsyncMock()
        mock_ws2.send_json = AsyncMock()

        await manager.connect(mock_ws1, user_id=1)
        await manager.connect(mock_ws2, user_id=2)

        message = {'type': 'broadcast', 'data': 'to all'}
        await manager.broadcast(message)

        mock_ws1.send_json.assert_called_once()
        mock_ws2.send_json.assert_called_once()

    def test_connection_count(self):
        """Test getting connection count."""
        from backend.websocket import ConnectionManager
        from unittest.mock import Mock

        manager = ConnectionManager()

        assert manager.get_connection_count() == 0

        # Manually add connections for testing
        manager.active_connections[1] = {Mock(), Mock()}
        manager.active_connections[2] = {Mock()}

        assert manager.get_connection_count() == 3

    @pytest.mark.asyncio
    async def test_keepalive_ping(self):
        """Test keepalive ping."""
        from backend.websocket import ConnectionManager

        manager = ConnectionManager()

        # Should not error even with no connections
        await manager.ping_all()
