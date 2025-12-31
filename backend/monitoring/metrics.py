# backend/monitoring/metrics.py

from typing import Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict
from backend.database import get_db
import time


class MetricsCollector:
    """Collect and aggregate system metrics."""

    def __init__(self):
        self.request_counts = defaultdict(int)
        self.response_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.start_time = time.time()

    def record_request(self, endpoint: str, method: str, status_code: int, duration_ms: float):
        """Record an API request."""
        key = f"{method} {endpoint}"
        self.request_counts[key] += 1
        self.response_times[key].append(duration_ms)

        if status_code >= 400:
            self.error_counts[key] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        total_requests = sum(self.request_counts.values())
        total_errors = sum(self.error_counts.values())

        avg_response_times = {}
        for endpoint, times in self.response_times.items():
            if times:
                avg_response_times[endpoint] = sum(times) / len(times)

        return {
            'uptime_seconds': int(time.time() - self.start_time),
            'total_requests': total_requests,
            'total_errors': total_errors,
            'error_rate': (total_errors / total_requests * 100) if total_requests > 0 else 0,
            'requests_by_endpoint': dict(self.request_counts),
            'avg_response_times': avg_response_times,
            'top_endpoints': sorted(
                self.request_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = get_db()
        cursor = conn.cursor()

        stats = {}

        # Count records in main tables
        tables = ['orders', 'users', 'positions', 'webhook_log', 'ab_tests']
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                result = cursor.fetchone()
                stats[f'{table}_count'] = result['count'] if result else 0
            except:
                stats[f'{table}_count'] = 0

        # Recent activity
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM orders
            WHERE created_at > datetime('now', '-24 hours')
        """)
        result = cursor.fetchone()
        stats['orders_last_24h'] = result['count'] if result else 0

        # Active users (logged in last 7 days)
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM users
            WHERE last_login > datetime('now', '-7 days')
        """)
        result = cursor.fetchone()
        stats['active_users_7d'] = result['count'] if result else 0

        conn.close()
        return stats

    def get_trading_stats(self) -> Dict[str, Any]:
        """Get trading statistics."""
        conn = get_db()
        cursor = conn.cursor()

        # Today's trades
        cursor.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN status = 'filled' THEN 1 ELSE 0 END) as filled_trades,
                SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected_trades
            FROM orders
            WHERE DATE(created_at) = DATE('now')
        """)
        today = cursor.fetchone()

        # This week's volume
        cursor.execute("""
            SELECT
                SUM(CASE WHEN qty IS NOT NULL THEN qty * filled_avg_price ELSE notional END) as volume
            FROM orders
            WHERE status = 'filled'
            AND created_at > datetime('now', '-7 days')
        """)
        volume = cursor.fetchone()

        conn.close()

        return {
            'today_total_trades': today['total_trades'] or 0,
            'today_filled_trades': today['filled_trades'] or 0,
            'today_rejected_trades': today['rejected_trades'] or 0,
            'week_volume': volume['volume'] or 0
        }


# Global metrics collector
_metrics_collector: MetricsCollector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    global _metrics_collector
    return _metrics_collector


def get_system_health() -> Dict[str, Any]:
    """Get overall system health status."""
    collector = get_metrics_collector()

    try:
        db_stats = collector.get_database_stats()
        trading_stats = collector.get_trading_stats()
        api_metrics = collector.get_metrics()

        # Determine health status
        error_rate = api_metrics['error_rate']

        if error_rate < 1:
            status = "healthy"
        elif error_rate < 5:
            status = "degraded"
        else:
            status = "unhealthy"

        return {
            'status': status,
            'timestamp': datetime.utcnow().isoformat(),
            'uptime_seconds': api_metrics['uptime_seconds'],
            'database': db_stats,
            'trading': trading_stats,
            'api': {
                'total_requests': api_metrics['total_requests'],
                'error_rate': round(error_rate, 2),
                'avg_response_time_ms': sum(api_metrics['avg_response_times'].values()) / len(api_metrics['avg_response_times']) if api_metrics['avg_response_times'] else 0
            }
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }
