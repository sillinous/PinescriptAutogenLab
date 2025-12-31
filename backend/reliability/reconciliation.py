# backend/reliability/reconciliation.py

"""
Order reconciliation system.

Ensures database state matches broker state by:
- Checking for orders stuck in 'pending' or 'submitted'
- Polling broker for order status updates
- Detecting missing fills
- Fixing discrepancies automatically
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from backend.database import get_db, update_order, insert_fill
from backend.brokers.alpaca_client import get_alpaca_client
from backend.monitoring.logger import log_info, log_error, log_warning
import asyncio


class OrderReconciliationService:
    """Service for reconciling orders with broker state."""

    def __init__(self):
        self.reconciliation_window_hours = 24  # Check orders from last 24 hours

    async def reconcile_all_orders(self) -> Dict[str, Any]:
        """
        Reconcile all pending/submitted orders with broker.

        Returns:
            Summary of reconciliation results
        """
        log_info("[RECONCILIATION] Starting order reconciliation")

        conn = get_db()
        cursor = conn.cursor()

        # Get orders that might need reconciliation
        cutoff_time = datetime.now() - timedelta(hours=self.reconciliation_window_hours)

        cursor.execute("""
            SELECT * FROM orders
            WHERE status IN ('pending', 'submitted', 'partial')
            AND created_at > ?
            ORDER BY created_at DESC
        """, (cutoff_time,))

        orders = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if not orders:
            log_info("[RECONCILIATION] No orders to reconcile")
            return {
                'total_checked': 0,
                'updated': 0,
                'errors': 0
            }

        log_info(f"[RECONCILIATION] Found {len(orders)} orders to check")

        results = {
            'total_checked': len(orders),
            'updated': 0,
            'errors': 0,
            'details': []
        }

        # Reconcile each order
        for order in orders:
            try:
                result = await self.reconcile_order(order)
                if result['updated']:
                    results['updated'] += 1
                results['details'].append(result)
            except Exception as e:
                results['errors'] += 1
                log_error(f"[RECONCILIATION] Error reconciling order {order['id']}: {e}")

        log_info(
            f"[RECONCILIATION] Complete. "
            f"Checked: {results['total_checked']}, "
            f"Updated: {results['updated']}, "
            f"Errors: {results['errors']}"
        )

        return results

    async def reconcile_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconcile a single order with broker.

        Args:
            order: Order record from database

        Returns:
            Reconciliation result
        """
        broker_type = order['broker_type']
        broker_order_id = order.get('broker_order_id')

        if not broker_order_id:
            # Order never made it to broker
            return {
                'order_id': order['id'],
                'status': 'no_broker_id',
                'updated': False,
                'message': 'Order has no broker_order_id'
            }

        # Currently only Alpaca supported
        if broker_type == 'alpaca':
            return await self._reconcile_alpaca_order(order)
        else:
            return {
                'order_id': order['id'],
                'status': 'unsupported_broker',
                'updated': False,
                'message': f'Broker {broker_type} not supported'
            }

    async def _reconcile_alpaca_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Reconcile order with Alpaca."""
        try:
            alpaca = get_alpaca_client()
            broker_order = await alpaca.get_order(order['broker_order_id'])

            if not broker_order:
                log_warning(f"[RECONCILIATION] Order {order['id']} not found at Alpaca")
                return {
                    'order_id': order['id'],
                    'status': 'not_found',
                    'updated': False,
                    'message': 'Order not found at broker'
                }

            # Check if status has changed
            broker_status = broker_order.get('status')
            current_status = order['status']

            # Map Alpaca status to internal status
            status_mapping = {
                'new': 'submitted',
                'accepted': 'submitted',
                'pending_new': 'submitted',
                'partially_filled': 'partial',
                'filled': 'filled',
                'done_for_day': 'filled',
                'canceled': 'cancelled',
                'expired': 'cancelled',
                'replaced': 'cancelled',
                'pending_cancel': 'cancelled',
                'pending_replace': 'submitted',
                'rejected': 'rejected',
                'suspended': 'rejected',
                'calculated': 'submitted'
            }

            mapped_status = status_mapping.get(broker_status, broker_status)

            # Update if status changed
            if mapped_status != current_status:
                update_data = {
                    'status': mapped_status,
                    'updated_at': datetime.now()
                }

                # Update fill data if available
                if broker_status in ['filled', 'partially_filled']:
                    filled_qty = float(broker_order.get('filled_qty', 0))
                    filled_avg_price = float(broker_order.get('filled_avg_price', 0))

                    if filled_qty > 0:
                        update_data['filled_qty'] = filled_qty
                        update_data['filled_avg_price'] = filled_avg_price
                        update_data['filled_at'] = datetime.now()

                        # Create fill record if doesn't exist
                        await self._ensure_fill_record(order['id'], filled_qty, filled_avg_price)

                update_order(order['id'], update_data)

                log_info(
                    f"[RECONCILIATION] Updated order {order['id']} from "
                    f"{current_status} to {mapped_status}"
                )

                return {
                    'order_id': order['id'],
                    'status': 'updated',
                    'updated': True,
                    'old_status': current_status,
                    'new_status': mapped_status,
                    'message': f'Status updated from {current_status} to {mapped_status}'
                }
            else:
                return {
                    'order_id': order['id'],
                    'status': 'unchanged',
                    'updated': False,
                    'message': 'Order status matches broker'
                }

        except Exception as e:
            log_error(f"[RECONCILIATION] Error reconciling Alpaca order {order['id']}: {e}")
            return {
                'order_id': order['id'],
                'status': 'error',
                'updated': False,
                'error': str(e)
            }

    async def _ensure_fill_record(self, order_id: int, qty: float, price: float):
        """Ensure fill record exists for order."""
        conn = get_db()
        cursor = conn.cursor()

        # Check if fill already exists
        cursor.execute("""
            SELECT id FROM fills WHERE order_id = ?
        """, (order_id,))

        if cursor.fetchone():
            conn.close()
            return  # Fill already exists

        conn.close()

        # Create fill record
        insert_fill(order_id, qty, price)
        log_info(f"[RECONCILIATION] Created fill record for order {order_id}")

    async def check_stale_orders(self) -> List[Dict[str, Any]]:
        """
        Find orders that have been in pending/submitted state too long.

        Returns:
            List of stale orders that may need manual intervention
        """
        conn = get_db()
        cursor = conn.cursor()

        # Orders pending for more than 1 hour
        stale_threshold = datetime.now() - timedelta(hours=1)

        cursor.execute("""
            SELECT * FROM orders
            WHERE status IN ('pending', 'submitted')
            AND created_at < ?
            ORDER BY created_at ASC
        """, (stale_threshold,))

        stale_orders = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if stale_orders:
            log_warning(f"[RECONCILIATION] Found {len(stale_orders)} stale orders")

        return stale_orders

    async def auto_fix_stale_orders(self):
        """
        Automatically fix stale orders by reconciling with broker.

        This should be run periodically (e.g., every 5 minutes) as a background task.
        """
        stale_orders = await self.check_stale_orders()

        if not stale_orders:
            return

        log_info(f"[RECONCILIATION] Auto-fixing {len(stale_orders)} stale orders")

        for order in stale_orders:
            try:
                await self.reconcile_order(order)
            except Exception as e:
                log_error(f"[RECONCILIATION] Failed to auto-fix order {order['id']}: {e}")


# Global reconciliation service
_reconciliation_service: Optional[OrderReconciliationService] = None


def get_reconciliation_service() -> OrderReconciliationService:
    """Get or create global reconciliation service instance."""
    global _reconciliation_service
    if _reconciliation_service is None:
        _reconciliation_service = OrderReconciliationService()
    return _reconciliation_service


# Background task runner
async def run_reconciliation_loop(interval_minutes: int = 5):
    """
    Run reconciliation loop in background.

    Args:
        interval_minutes: How often to run reconciliation (default: 5 minutes)
    """
    service = get_reconciliation_service()
    log_info(f"[RECONCILIATION] Starting background reconciliation loop (every {interval_minutes} min)")

    while True:
        try:
            await service.auto_fix_stale_orders()
        except Exception as e:
            log_error(f"[RECONCILIATION] Background loop error: {e}")

        await asyncio.sleep(interval_minutes * 60)
