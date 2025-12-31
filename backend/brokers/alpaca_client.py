# backend/brokers/alpaca_client.py

import httpx
from typing import Dict, Any, Optional, List
from datetime import datetime
from backend.config import Config
import json


class AlpacaClient:
    """Alpaca API client for stock/ETF trading."""

    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = None):
        self.api_key = api_key or Config.ALPACA_API_KEY
        self.secret_key = secret_key or Config.ALPACA_SECRET_KEY
        self.paper = paper if paper is not None else Config.ALPACA_PAPER_TRADING

        self.base_url = (
            "https://paper-api.alpaca.markets" if self.paper
            else "https://api.alpaca.markets"
        )

        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json"
        }

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not configured")

    async def get_account(self) -> Dict[str, Any]:
        """Get account information."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/v2/account",
                headers=self.headers,
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/v2/positions",
                headers=self.headers,
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()

    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for specific symbol."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/v2/positions/{symbol}",
                    headers=self.headers,
                    timeout=10.0
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def submit_order(
        self,
        symbol: str,
        side: str,
        qty: Optional[float] = None,
        notional: Optional[float] = None,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit order to Alpaca.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            side: 'buy' or 'sell'
            qty: Number of shares (mutually exclusive with notional)
            notional: Dollar amount (mutually exclusive with qty)
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
            limit_price: Required for limit orders
            client_order_id: Unique client order ID
        """
        order_data = {
            "symbol": symbol.upper(),
            "side": side.lower(),
            "type": order_type,
            "time_in_force": time_in_force
        }

        # Qty or notional (not both)
        if qty is not None:
            order_data["qty"] = qty
        elif notional is not None:
            order_data["notional"] = notional
        else:
            raise ValueError("Must specify either qty or notional")

        if limit_price is not None:
            order_data["limit_price"] = limit_price

        if client_order_id:
            order_data["client_order_id"] = client_order_id

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v2/orders",
                headers=self.headers,
                json=order_data,
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()

    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get order status by Alpaca order ID."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/v2/orders/{order_id}",
                headers=self.headers,
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()

    async def get_orders(self, status: str = "all", limit: int = 100) -> List[Dict[str, Any]]:
        """Get orders with optional status filter."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/v2/orders",
                headers=self.headers,
                params={"status": status, "limit": limit},
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an open order."""
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{self.base_url}/v2/orders/{order_id}",
                headers=self.headers,
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()

    async def get_bars(self, symbol: str, timeframe: str = "1Day", limit: int = 100) -> Dict[str, Any]:
        """Get historical price bars."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/v2/stocks/{symbol}/bars",
                headers=self.headers,
                params={"timeframe": timeframe, "limit": limit},
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()

    def map_to_internal_order(self, alpaca_order: Dict[str, Any]) -> Dict[str, Any]:
        """Map Alpaca order response to internal format."""
        return {
            'broker_order_id': alpaca_order['id'],
            'status': self._map_status(alpaca_order['status']),
            'filled_qty': float(alpaca_order.get('filled_qty', 0)),
            'filled_avg_price': float(alpaca_order.get('filled_avg_price', 0)) if alpaca_order.get('filled_avg_price') else None,
            'submitted_at': alpaca_order.get('submitted_at'),
            'filled_at': alpaca_order.get('filled_at'),
            'updated_at': alpaca_order.get('updated_at')
        }

    def _map_status(self, alpaca_status: str) -> str:
        """Map Alpaca order status to internal status."""
        status_map = {
            'new': 'submitted',
            'accepted': 'submitted',
            'pending_new': 'pending',
            'accepted_for_bidding': 'submitted',
            'stopped': 'submitted',
            'partially_filled': 'partial',
            'filled': 'filled',
            'done_for_day': 'filled',
            'canceled': 'cancelled',
            'expired': 'cancelled',
            'replaced': 'cancelled',
            'pending_cancel': 'cancelled',
            'pending_replace': 'submitted',
            'rejected': 'rejected',
            'suspended': 'rejected'
        }
        return status_map.get(alpaca_status, 'pending')


# Singleton instance (can be overridden with custom credentials)
_alpaca_instance: Optional[AlpacaClient] = None


def get_alpaca_client(api_key: str = None, secret_key: str = None, paper: bool = None) -> AlpacaClient:
    """Get or create Alpaca client instance."""
    global _alpaca_instance

    if api_key or secret_key or paper is not None:
        # Create new instance with custom credentials
        return AlpacaClient(api_key, secret_key, paper)

    if _alpaca_instance is None:
        _alpaca_instance = AlpacaClient()

    return _alpaca_instance
