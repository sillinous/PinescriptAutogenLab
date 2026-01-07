# backend/order_service.py

from typing import Dict, Any, Optional
from datetime import datetime
import uuid
import json
import os
import logging
from backend.database import (
    insert_order, update_order, get_order_by_id,
    insert_fill, upsert_position, log_webhook,
    get_order_by_client_order_id
)
from backend.brokers.alpaca_client import get_alpaca_client
import asyncio

try:
    import ccxt.async_support as ccxt_async
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    ccxt_async = None

logger = logging.getLogger(__name__)


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
        """
        Execute crypto order via CCXT.

        Supported exchanges: binance, coinbase, kraken, bybit, okx
        """
        if not CCXT_AVAILABLE:
            raise RuntimeError("CCXT library not available. Install with: pip install ccxt")

        # Extract order parameters
        exchange_id = payload.get('exchange', 'binance').lower()
        symbol = payload.get('symbol')  # Format: "BTC/USDT"
        side = payload.get('side', 'buy').lower()
        qty = payload.get('qty')
        order_type = payload.get('type', 'market').lower()
        limit_price = payload.get('limit_price')
        time_in_force = payload.get('time_in_force')

        if not symbol:
            raise ValueError("Missing required field: symbol (format: BTC/USDT)")
        if side not in ['buy', 'sell']:
            raise ValueError(f"Invalid side: {side}")
        if not qty:
            raise ValueError("Missing required field: qty")

        # Normalize symbol format (e.g., BTC_USDT -> BTC/USDT)
        if '_' in symbol:
            symbol = symbol.replace('_', '/')

        # Generate client order ID
        client_order_id = f"pinelab_{uuid.uuid4().hex[:16]}"

        # Insert order to database (pending)
        order_id = insert_order({
            'client_order_id': client_order_id,
            'broker_type': f'ccxt_{exchange_id}',
            'symbol': symbol,
            'side': side,
            'order_type': order_type,
            'qty': qty,
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

        exchange = None
        try:
            # Create exchange instance
            exchange = await self._get_ccxt_exchange(exchange_id)

            # Check if exchange is enabled for trading
            if not exchange.has.get('createOrder'):
                raise RuntimeError(f"Exchange {exchange_id} does not support order creation")

            # Build order parameters
            order_params = {}
            if time_in_force:
                order_params['timeInForce'] = time_in_force.upper()

            # Submit order
            if order_type == 'market':
                ccxt_order = await exchange.create_market_order(
                    symbol=symbol,
                    side=side,
                    amount=float(qty),
                    params=order_params
                )
            elif order_type == 'limit':
                if not limit_price:
                    raise ValueError("Limit orders require limit_price")
                ccxt_order = await exchange.create_limit_order(
                    symbol=symbol,
                    side=side,
                    amount=float(qty),
                    price=float(limit_price),
                    params=order_params
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            # Map CCXT response to internal format
            mapped_order = self._map_ccxt_order(ccxt_order, exchange_id)
            update_order(order_id, mapped_order)

            # If filled immediately, record fill
            if mapped_order['status'] == 'filled':
                await self._process_ccxt_fill(order_id, ccxt_order, exchange_id)

            logger.info(f"CCXT order submitted: {client_order_id} on {exchange_id}")

            return {
                'success': True,
                'order_id': order_id,
                'broker_order_id': ccxt_order.get('id'),
                'status': mapped_order['status'],
                'exchange': exchange_id,
                'message': f"Order {client_order_id} submitted to {exchange_id}"
            }

        except Exception as e:
            logger.error(f"CCXT order failed: {e}")
            update_order(order_id, {
                'status': 'rejected',
                'error_message': str(e)
            })
            raise

        finally:
            if exchange:
                await exchange.close()

    async def _get_ccxt_exchange(self, exchange_id: str):
        """
        Create and configure CCXT exchange instance.

        Credentials are loaded from environment variables:
        - {EXCHANGE}_API_KEY
        - {EXCHANGE}_SECRET
        - {EXCHANGE}_PASSWORD (if required)
        """
        if not CCXT_AVAILABLE:
            raise RuntimeError("CCXT not available")

        exchange_class = getattr(ccxt_async, exchange_id, None)
        if not exchange_class:
            raise ValueError(f"Unsupported exchange: {exchange_id}")

        # Load credentials from environment
        prefix = exchange_id.upper()
        api_key = os.getenv(f'{prefix}_API_KEY')
        secret = os.getenv(f'{prefix}_SECRET')
        password = os.getenv(f'{prefix}_PASSWORD')  # For exchanges like KuCoin

        config = {
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # Can be 'spot', 'future', 'margin'
            }
        }

        if api_key and secret:
            config['apiKey'] = api_key
            config['secret'] = secret
            if password:
                config['password'] = password

        # Check for testnet/sandbox mode
        use_sandbox = os.getenv(f'{prefix}_SANDBOX', 'false').lower() == 'true'
        if use_sandbox:
            config['sandbox'] = True

        exchange = exchange_class(config)

        # Load markets
        await exchange.load_markets()

        return exchange

    def _map_ccxt_order(self, ccxt_order: Dict[str, Any], exchange_id: str) -> Dict[str, Any]:
        """Map CCXT order response to internal format."""
        status_map = {
            'open': 'open',
            'closed': 'filled',
            'canceled': 'canceled',
            'expired': 'expired',
            'rejected': 'rejected'
        }

        ccxt_status = ccxt_order.get('status', 'open')

        return {
            'broker_order_id': ccxt_order.get('id'),
            'status': status_map.get(ccxt_status, ccxt_status),
            'filled_qty': ccxt_order.get('filled', 0),
            'filled_avg_price': ccxt_order.get('average'),
            'broker_response': json.dumps(ccxt_order)
        }

    async def _process_ccxt_fill(
        self,
        order_id: int,
        ccxt_order: Dict[str, Any],
        exchange_id: str
    ):
        """Process filled CCXT order and update positions."""
        filled_qty = float(ccxt_order.get('filled', 0))
        filled_price = ccxt_order.get('average') or ccxt_order.get('price', 0)
        if filled_price:
            filled_price = float(filled_price)
        symbol = ccxt_order['symbol']
        side = ccxt_order['side']

        if filled_qty <= 0 or not filled_price:
            return

        # Record fill
        insert_fill(
            order_id=order_id,
            qty=filled_qty,
            price=filled_price,
            broker_fill_id=ccxt_order.get('id')
        )

        # Update position
        exchange = None
        try:
            exchange = await self._get_ccxt_exchange(exchange_id)

            # Fetch balance to get position
            balance = await exchange.fetch_balance()

            # Extract base currency from symbol (e.g., BTC from BTC/USDT)
            base_currency = symbol.split('/')[0]

            if base_currency in balance.get('total', {}):
                qty = float(balance['total'][base_currency])
                free = float(balance['free'].get(base_currency, 0))

                upsert_position(
                    broker_type=f'ccxt_{exchange_id}',
                    symbol=symbol,
                    qty=qty,
                    avg_entry_price=filled_price,  # Approximation
                    current_price=filled_price,
                    unrealized_pnl=0  # CCXT doesn't provide this directly
                )

        except Exception as e:
            logger.warning(f"Failed to update position for {symbol}: {e}")

        finally:
            if exchange:
                await exchange.close()

    async def sync_ccxt_positions(self, exchange_id: str = 'binance'):
        """Sync positions from a CCXT exchange."""
        exchange = None
        try:
            exchange = await self._get_ccxt_exchange(exchange_id)
            balance = await exchange.fetch_balance()

            positions_updated = 0
            for currency, amount in balance.get('total', {}).items():
                if float(amount) > 0:
                    # Skip stable coins and fiat
                    if currency in ['USD', 'USDT', 'USDC', 'BUSD', 'EUR']:
                        continue

                    upsert_position(
                        broker_type=f'ccxt_{exchange_id}',
                        symbol=f'{currency}/USDT',  # Approximate
                        qty=float(amount),
                        avg_entry_price=0,  # Not available from balance
                        current_price=0,
                        unrealized_pnl=0
                    )
                    positions_updated += 1

            logger.info(f"Synced {positions_updated} positions from {exchange_id}")
            return positions_updated

        except Exception as e:
            logger.error(f"Failed to sync {exchange_id} positions: {e}")
            raise

        finally:
            if exchange:
                await exchange.close()

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

            synced_count = 0
            for alpaca_order in alpaca_orders:
                # Find matching order by client_order_id
                client_order_id = alpaca_order.get('client_order_id')
                if not client_order_id or not client_order_id.startswith('pinelab_'):
                    continue

                # Find local order
                local_order = get_order_by_client_order_id(client_order_id)
                if not local_order:
                    continue

                # Update order status
                mapped_order = alpaca.map_to_internal_order(alpaca_order)
                update_order(local_order['id'], mapped_order)
                synced_count += 1

                # Process fill if completed
                if mapped_order['status'] == 'filled' and local_order['status'] != 'filled':
                    await self._process_alpaca_fill(local_order['id'], alpaca_order)

            logger.info(f"Synced {synced_count} orders from Alpaca")
            return alpaca_orders

        except Exception as e:
            logger.error(f"Failed to sync Alpaca orders: {e}")
            raise


# Singleton service
_order_service_instance = OrderExecutionService()


def get_order_service() -> OrderExecutionService:
    """Get order execution service instance."""
    return _order_service_instance
