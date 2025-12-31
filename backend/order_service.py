# backend/order_service.py

from typing import Dict, Any, Optional
from datetime import datetime
import uuid
import json
from backend.database import (
    insert_order, update_order, get_order_by_id,
    insert_fill, upsert_position, log_webhook
)
from backend.brokers.alpaca_client import get_alpaca_client
import asyncio


class OrderExecutionService:
    """Service for executing and tracking orders across brokers."""

    async def execute_webhook_order(
        self,
        payload: Dict[str, Any],
        webhook_source: str = "tradingview",
        ip_address: str = None
    ) -> Dict[str, Any]:
        """
        Process webhook and execute order.

        Expected payload formats:

        Equities (Alpaca):
        {
            "ticker": "AAPL",
            "side": "buy",
            "qty": 10,          # OR
            "notional": 1000,   # dollar amount
            "type": "market"    # optional
        }

        Crypto (CCXT):
        {
            "market": "crypto",
            "exchange": "binance",
            "symbol": "BTC/USDT",
            "side": "buy",
            "qty": 0.001
        }
        """
        try:
            # Determine broker type
            if payload.get('market') == 'crypto':
                return await self._execute_crypto_order(payload, webhook_source, ip_address)
            else:
                return await self._execute_alpaca_order(payload, webhook_source, ip_address)

        except Exception as e:
            error_msg = f"Order execution failed: {str(e)}"
            log_webhook(
                payload=json.dumps(payload),
                status='rejected',
                error_message=error_msg,
                source=webhook_source,
                ip_address=ip_address
            )
            raise

    async def _execute_alpaca_order(
        self,
        payload: Dict[str, Any],
        webhook_source: str,
        ip_address: str
    ) -> Dict[str, Any]:
        """Execute equity order via Alpaca."""

        # Extract order parameters
        symbol = payload.get('ticker') or payload.get('symbol')
        side = payload.get('side', 'buy').lower()
        qty = payload.get('qty')
        notional = payload.get('notional')
        order_type = payload.get('type', 'market').lower()
        limit_price = payload.get('limit_price')
        time_in_force = payload.get('time_in_force', 'day')

        if not symbol:
            raise ValueError("Missing required field: ticker/symbol")
        if side not in ['buy', 'sell']:
            raise ValueError(f"Invalid side: {side}")
        if not qty and not notional:
            raise ValueError("Must specify either qty or notional")

        # Generate client order ID
        client_order_id = f"pinelab_{uuid.uuid4().hex[:16]}"

        # Insert order to database (pending)
        order_id = insert_order({
            'client_order_id': client_order_id,
            'broker_type': 'alpaca',
            'symbol': symbol.upper(),
            'side': side,
            'order_type': order_type,
            'qty': qty,
            'notional': notional,
            'limit_price': limit_price,
            'time_in_force': time_in_force,
            'webhook_payload': json.dumps(payload),
            'status': 'pending'
        })

        # Log webhook
        log_webhook(
            payload=json.dumps(payload),
            source=webhook_source,
            ip_address=ip_address,
            order_id=order_id,
            status='processed'
        )

        try:
            # Submit to Alpaca
            alpaca = get_alpaca_client()
            alpaca_response = await alpaca.submit_order(
                symbol=symbol,
                side=side,
                qty=qty,
                notional=notional,
                order_type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                client_order_id=client_order_id
            )

            # Update order with broker response
            mapped_order = alpaca.map_to_internal_order(alpaca_response)
            update_order(order_id, mapped_order)

            # If filled immediately, record fill
            if mapped_order['status'] == 'filled':
                await self._process_alpaca_fill(order_id, alpaca_response)

            return {
                'success': True,
                'order_id': order_id,
                'broker_order_id': alpaca_response['id'],
                'status': mapped_order['status'],
                'message': f"Order {client_order_id} submitted successfully"
            }

        except Exception as e:
            # Update order with error
            update_order(order_id, {
                'status': 'rejected',
                'error_message': str(e)
            })
            raise

    async def _execute_crypto_order(
        self,
        payload: Dict[str, Any],
        webhook_source: str,
        ip_address: str
    ) -> Dict[str, Any]:
        """Execute crypto order via CCXT (to be implemented)."""
        # TODO: Implement CCXT integration
        raise NotImplementedError("Crypto trading via CCXT not yet implemented")

    async def _process_alpaca_fill(self, order_id: int, alpaca_order: Dict[str, Any]):
        """Process filled order and update positions."""
        filled_qty = float(alpaca_order.get('filled_qty', 0))
        filled_price = float(alpaca_order.get('filled_avg_price', 0))
        symbol = alpaca_order['symbol']
        side = alpaca_order['side']

        if filled_qty <= 0 or filled_price <= 0:
            return

        # Record fill
        insert_fill(
            order_id=order_id,
            qty=filled_qty,
            price=filled_price,
            broker_fill_id=alpaca_order['id']
        )

        # Update position
        try:
            alpaca = get_alpaca_client()
            position = await alpaca.get_position(symbol)

            if position:
                upsert_position(
                    broker_type='alpaca',
                    symbol=symbol,
                    qty=float(position['qty']),
                    avg_entry_price=float(position['avg_entry_price']),
                    current_price=float(position['current_price']),
                    unrealized_pnl=float(position['unrealized_pl'])
                )
        except Exception as e:
            print(f"[WARN] Failed to update position for {symbol}: {e}")

    async def sync_alpaca_positions(self):
        """Sync all positions from Alpaca to database."""
        try:
            alpaca = get_alpaca_client()
            positions = await alpaca.get_positions()

            for pos in positions:
                upsert_position(
                    broker_type='alpaca',
                    symbol=pos['symbol'],
                    qty=float(pos['qty']),
                    avg_entry_price=float(pos['avg_entry_price']),
                    current_price=float(pos['current_price']),
                    unrealized_pnl=float(pos['unrealized_pl'])
                )

            print(f"[OK] Synced {len(positions)} positions from Alpaca")
            return positions

        except Exception as e:
            print(f"[ERROR] Failed to sync Alpaca positions: {e}")
            raise

    async def sync_alpaca_orders(self, limit: int = 100):
        """Sync recent orders from Alpaca."""
        try:
            alpaca = get_alpaca_client()
            alpaca_orders = await alpaca.get_orders(status='all', limit=limit)

            for alpaca_order in alpaca_orders:
                # Find matching order by client_order_id
                client_order_id = alpaca_order.get('client_order_id')
                if not client_order_id or not client_order_id.startswith('pinelab_'):
                    continue

                # Update order status
                mapped_order = alpaca.map_to_internal_order(alpaca_order)
                # This requires a helper to find order by client_order_id - TODO

            print(f"[OK] Synced {len(alpaca_orders)} orders from Alpaca")
            return alpaca_orders

        except Exception as e:
            print(f"[ERROR] Failed to sync Alpaca orders: {e}")
            raise


# Singleton service
_order_service_instance = OrderExecutionService()


def get_order_service() -> OrderExecutionService:
    """Get order execution service instance."""
    return _order_service_instance
