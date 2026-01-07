# backend/websocket/price_streamer.py
"""
Real-time price streaming service via WebSocket.

Features:
- Subscribe to real-time price updates for crypto pairs
- Aggregate multiple data sources (Crypto.com, CCXT exchanges)
- Efficient broadcasting to multiple clients
- Automatic reconnection and error handling
- Rate limiting and throttling
"""

import asyncio
import json
import logging
from typing import Dict, Set, Any, Optional, List
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
import httpx

logger = logging.getLogger(__name__)


class PriceSubscription:
    """Tracks subscriptions for a symbol."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.subscribers: Set[WebSocket] = set()
        self.last_price: Optional[float] = None
        self.last_update: Optional[datetime] = None
        self.bid: Optional[float] = None
        self.ask: Optional[float] = None
        self.volume_24h: Optional[float] = None
        self.change_24h: Optional[float] = None


class PriceStreamer:
    """
    Manages real-time price streaming to WebSocket clients.

    Usage:
        streamer = PriceStreamer()
        await streamer.start()

        # In WebSocket endpoint:
        await streamer.subscribe(websocket, "BTC_USDT")
        await streamer.unsubscribe(websocket, "BTC_USDT")
    """

    # Crypto.com API base URL
    CRYPTO_API_BASE = "https://api.crypto.com/exchange/v1"

    # Default symbols to track
    DEFAULT_SYMBOLS = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "XRP_USDT", "ADA_USDT"]

    # Update interval in seconds
    UPDATE_INTERVAL = 1.0

    def __init__(self):
        self.subscriptions: Dict[str, PriceSubscription] = {}
        self.all_subscribers: Set[WebSocket] = set()  # Subscribers to all prices
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        self._http_client: Optional[httpx.AsyncClient] = None

    async def start(self):
        """Start the price streaming service."""
        if self._running:
            return

        self._running = True
        self._http_client = httpx.AsyncClient(timeout=10.0)

        # Initialize default symbol subscriptions
        for symbol in self.DEFAULT_SYMBOLS:
            self.subscriptions[symbol] = PriceSubscription(symbol)

        # Start update loop
        self._update_task = asyncio.create_task(self._price_update_loop())
        logger.info("[PRICE_STREAMER] Started price streaming service")

    async def stop(self):
        """Stop the price streaming service."""
        self._running = False

        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        if self._http_client:
            await self._http_client.aclose()

        # Notify all subscribers
        await self._broadcast_all({
            "type": "system",
            "event": "stream_stopped",
            "message": "Price streaming service stopped"
        })

        logger.info("[PRICE_STREAMER] Stopped price streaming service")

    async def subscribe(self, websocket: WebSocket, symbol: Optional[str] = None):
        """
        Subscribe a WebSocket to price updates.

        Args:
            websocket: WebSocket connection
            symbol: Symbol to subscribe to (None = all symbols)
        """
        if symbol:
            # Normalize symbol format
            symbol = symbol.upper().replace("/", "_")

            if symbol not in self.subscriptions:
                self.subscriptions[symbol] = PriceSubscription(symbol)

            self.subscriptions[symbol].subscribers.add(websocket)
            logger.info(f"[PRICE_STREAMER] Client subscribed to {symbol}")

            # Send current price immediately if available
            sub = self.subscriptions[symbol]
            if sub.last_price:
                await self._send_price_update(websocket, symbol, sub)
        else:
            # Subscribe to all symbols
            self.all_subscribers.add(websocket)
            logger.info("[PRICE_STREAMER] Client subscribed to all symbols")

            # Send all current prices
            await self._send_all_prices(websocket)

    async def unsubscribe(self, websocket: WebSocket, symbol: Optional[str] = None):
        """
        Unsubscribe a WebSocket from price updates.

        Args:
            websocket: WebSocket connection
            symbol: Symbol to unsubscribe from (None = all symbols)
        """
        if symbol:
            symbol = symbol.upper().replace("/", "_")
            if symbol in self.subscriptions:
                self.subscriptions[symbol].subscribers.discard(websocket)
        else:
            self.all_subscribers.discard(websocket)
            # Also remove from individual subscriptions
            for sub in self.subscriptions.values():
                sub.subscribers.discard(websocket)

    async def disconnect(self, websocket: WebSocket):
        """Remove WebSocket from all subscriptions."""
        self.all_subscribers.discard(websocket)
        for sub in self.subscriptions.values():
            sub.subscribers.discard(websocket)

    async def _price_update_loop(self):
        """Main loop for fetching and broadcasting prices."""
        while self._running:
            try:
                # Fetch prices for all subscribed symbols
                symbols = list(self.subscriptions.keys())

                if symbols:
                    prices = await self._fetch_prices(symbols)

                    for symbol, price_data in prices.items():
                        if symbol in self.subscriptions:
                            sub = self.subscriptions[symbol]

                            # Update subscription data
                            sub.last_price = price_data.get("price")
                            sub.bid = price_data.get("bid")
                            sub.ask = price_data.get("ask")
                            sub.volume_24h = price_data.get("volume_24h")
                            sub.change_24h = price_data.get("change_24h")
                            sub.last_update = datetime.now()

                            # Broadcast to subscribers
                            await self._broadcast_price(symbol, sub)

            except Exception as e:
                logger.error(f"[PRICE_STREAMER] Update loop error: {e}")

            await asyncio.sleep(self.UPDATE_INTERVAL)

    async def _fetch_prices(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch current prices from Crypto.com API.

        Returns dict of symbol -> price data
        """
        prices = {}

        try:
            # Use ticker endpoint for multiple symbols
            response = await self._http_client.get(
                f"{self.CRYPTO_API_BASE}/public/get-tickers"
            )

            if response.status_code == 200:
                data = response.json()
                result = data.get("result", {})
                tickers = result.get("data", [])

                for ticker in tickers:
                    symbol = ticker.get("i", "").replace("-", "_")

                    if symbol in symbols:
                        prices[symbol] = {
                            "price": float(ticker.get("a", 0)),  # Last trade price
                            "bid": float(ticker.get("b", 0)),    # Best bid
                            "ask": float(ticker.get("k", 0)),    # Best ask
                            "volume_24h": float(ticker.get("v", 0)),
                            "change_24h": float(ticker.get("c", 0)),  # 24h change %
                            "high_24h": float(ticker.get("h", 0)),
                            "low_24h": float(ticker.get("l", 0)),
                        }

        except Exception as e:
            logger.error(f"[PRICE_STREAMER] Failed to fetch prices: {e}")

        return prices

    async def _broadcast_price(self, symbol: str, sub: PriceSubscription):
        """Broadcast price update to all subscribers of a symbol."""
        message = {
            "type": "price",
            "symbol": symbol,
            "price": sub.last_price,
            "bid": sub.bid,
            "ask": sub.ask,
            "volume_24h": sub.volume_24h,
            "change_24h": sub.change_24h,
            "timestamp": sub.last_update.isoformat() if sub.last_update else None
        }

        # Send to symbol-specific subscribers
        disconnected = []
        for ws in sub.subscribers:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)

        # Send to all-symbols subscribers
        for ws in self.all_subscribers:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)

        # Clean up disconnected clients
        for ws in disconnected:
            await self.disconnect(ws)

    async def _broadcast_all(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        disconnected = []

        all_clients = self.all_subscribers.copy()
        for sub in self.subscriptions.values():
            all_clients.update(sub.subscribers)

        for ws in all_clients:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            await self.disconnect(ws)

    async def _send_price_update(self, websocket: WebSocket, symbol: str, sub: PriceSubscription):
        """Send current price to a specific client."""
        try:
            await websocket.send_json({
                "type": "price",
                "symbol": symbol,
                "price": sub.last_price,
                "bid": sub.bid,
                "ask": sub.ask,
                "volume_24h": sub.volume_24h,
                "change_24h": sub.change_24h,
                "timestamp": sub.last_update.isoformat() if sub.last_update else None
            })
        except Exception as e:
            logger.warning(f"[PRICE_STREAMER] Failed to send price to client: {e}")

    async def _send_all_prices(self, websocket: WebSocket):
        """Send all current prices to a client."""
        prices = []
        for symbol, sub in self.subscriptions.items():
            if sub.last_price:
                prices.append({
                    "symbol": symbol,
                    "price": sub.last_price,
                    "bid": sub.bid,
                    "ask": sub.ask,
                    "volume_24h": sub.volume_24h,
                    "change_24h": sub.change_24h,
                    "timestamp": sub.last_update.isoformat() if sub.last_update else None
                })

        try:
            await websocket.send_json({
                "type": "prices",
                "data": prices,
                "count": len(prices)
            })
        except Exception as e:
            logger.warning(f"[PRICE_STREAMER] Failed to send all prices: {e}")

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get the current cached price for a symbol."""
        symbol = symbol.upper().replace("/", "_")
        if symbol in self.subscriptions:
            return self.subscriptions[symbol].last_price
        return None

    def get_all_prices(self) -> Dict[str, float]:
        """Get all current cached prices."""
        return {
            symbol: sub.last_price
            for symbol, sub in self.subscriptions.items()
            if sub.last_price is not None
        }

    def get_subscriber_count(self) -> int:
        """Get total number of unique subscribers."""
        all_subs = self.all_subscribers.copy()
        for sub in self.subscriptions.values():
            all_subs.update(sub.subscribers)
        return len(all_subs)


# Singleton instance
_price_streamer: Optional[PriceStreamer] = None


def get_price_streamer() -> PriceStreamer:
    """Get or create the price streamer instance."""
    global _price_streamer
    if _price_streamer is None:
        _price_streamer = PriceStreamer()
    return _price_streamer


async def start_price_streamer():
    """Start the global price streamer."""
    streamer = get_price_streamer()
    await streamer.start()


async def stop_price_streamer():
    """Stop the global price streamer."""
    global _price_streamer
    if _price_streamer:
        await _price_streamer.stop()
        _price_streamer = None
