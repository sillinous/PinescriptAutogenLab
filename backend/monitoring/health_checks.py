# backend/monitoring/health_checks.py

"""
Enhanced health check system for monitoring all dependencies.

Checks:
- Database connectivity and performance
- Broker API connectivity (Alpaca)
- SMTP connectivity (if enabled)
- Disk space
- Memory usage
- System resources
"""

import os
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import psutil


class HealthCheckService:
    """Comprehensive health checking for all system dependencies."""

    def __init__(self):
        self.start_time = datetime.now()

    async def check_all(self, include_external: bool = True) -> Dict[str, Any]:
        """
        Run all health checks.

        Args:
            include_external: Whether to check external dependencies (broker, SMTP)

        Returns:
            Health check results
        """
        results = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'checks': {}
        }

        # Internal checks (always run)
        results['checks']['database'] = await self.check_database()
        results['checks']['disk'] = self.check_disk_space()
        results['checks']['memory'] = self.check_memory()
        results['checks']['config'] = self.check_config()

        # External checks (optional, can be slow)
        if include_external:
            results['checks']['broker'] = await self.check_broker()
            results['checks']['smtp'] = await self.check_smtp()

        # Determine overall status
        unhealthy_checks = [
            name for name, check in results['checks'].items()
            if check.get('status') != 'healthy'
        ]

        if unhealthy_checks:
            results['status'] = 'degraded'
            results['unhealthy_checks'] = unhealthy_checks

        # Count checks
        results['total_checks'] = len(results['checks'])
        results['healthy_checks'] = sum(
            1 for check in results['checks'].values()
            if check.get('status') == 'healthy'
        )

        return results

    async def check_database(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            from backend.database import get_db

            start = time.time()

            # Test connection
            conn = get_db()
            cursor = conn.cursor()

            # Simple query
            cursor.execute("SELECT 1")
            result = cursor.fetchone()

            # Check some table counts
            cursor.execute("SELECT COUNT(*) as count FROM orders")
            orders_count = cursor.fetchone()['count']

            cursor.execute("SELECT COUNT(*) as count FROM users")
            users_count = cursor.fetchone()['count']

            conn.close()

            duration_ms = (time.time() - start) * 1000

            # Check database file size
            from backend.config import Config
            data_dir = Path(os.getenv("PINELAB_DATA", "./data"))
            db_path = data_dir / "pinelab.db"

            db_size_mb = 0
            if db_path.exists():
                db_size_mb = db_path.stat().st_size / (1024 * 1024)

            return {
                'status': 'healthy',
                'response_time_ms': round(duration_ms, 2),
                'orders_count': orders_count,
                'users_count': users_count,
                'database_size_mb': round(db_size_mb, 2),
                'message': 'Database is responsive'
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'message': 'Database connection failed'
            }

    async def check_broker(self) -> Dict[str, Any]:
        """Check broker API connectivity."""
        try:
            from backend.brokers.alpaca_client import get_alpaca_client

            start = time.time()

            alpaca = get_alpaca_client()

            # Test API call
            account = await alpaca.get_account()

            duration_ms = (time.time() - start) * 1000

            return {
                'status': 'healthy',
                'broker': 'alpaca',
                'response_time_ms': round(duration_ms, 2),
                'account_status': account.get('status', 'unknown'),
                'buying_power': float(account.get('buying_power', 0)),
                'message': 'Broker API is responsive'
            }

        except ValueError as e:
            # Broker not configured
            return {
                'status': 'not_configured',
                'message': str(e)
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'message': 'Broker API connection failed'
            }

    async def check_smtp(self) -> Dict[str, Any]:
        """Check SMTP connectivity."""
        smtp_user = os.getenv('SMTP_USER')
        smtp_password = os.getenv('SMTP_PASSWORD')

        if not smtp_user or not smtp_password:
            return {
                'status': 'not_configured',
                'message': 'SMTP not configured (optional)'
            }

        try:
            import smtplib

            smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
            smtp_port = int(os.getenv('SMTP_PORT', '587'))

            start = time.time()

            with smtplib.SMTP(smtp_host, smtp_port, timeout=5) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)

            duration_ms = (time.time() - start) * 1000

            return {
                'status': 'healthy',
                'response_time_ms': round(duration_ms, 2),
                'host': smtp_host,
                'port': smtp_port,
                'message': 'SMTP connection successful'
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'message': 'SMTP connection failed'
            }

    def check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            data_dir = Path(os.getenv("PINELAB_DATA", "./data"))
            data_dir.mkdir(exist_ok=True)

            # Get disk usage for data directory
            usage = psutil.disk_usage(str(data_dir))

            percent_used = usage.percent
            free_gb = usage.free / (1024 ** 3)

            # Determine status
            if percent_used > 95:
                status = 'critical'
                message = f'Disk critically low ({free_gb:.1f} GB free)'
            elif percent_used > 90:
                status = 'warning'
                message = f'Disk space low ({free_gb:.1f} GB free)'
            else:
                status = 'healthy'
                message = f'Sufficient disk space ({free_gb:.1f} GB free)'

            return {
                'status': status,
                'total_gb': round(usage.total / (1024 ** 3), 2),
                'used_gb': round(usage.used / (1024 ** 3), 2),
                'free_gb': round(free_gb, 2),
                'percent_used': round(percent_used, 1),
                'message': message
            }

        except Exception as e:
            return {
                'status': 'unknown',
                'error': str(e),
                'message': 'Could not check disk space'
            }

    def check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory_mb = process.memory_info().rss / (1024 * 1024)

            percent_used = memory.percent
            available_gb = memory.available / (1024 ** 3)

            # Determine status
            if percent_used > 95:
                status = 'critical'
                message = f'Memory critically low ({available_gb:.1f} GB available)'
            elif percent_used > 90:
                status = 'warning'
                message = f'Memory usage high ({available_gb:.1f} GB available)'
            else:
                status = 'healthy'
                message = f'Memory usage normal ({available_gb:.1f} GB available)'

            return {
                'status': status,
                'total_gb': round(memory.total / (1024 ** 3), 2),
                'available_gb': round(available_gb, 2),
                'percent_used': round(percent_used, 1),
                'process_memory_mb': round(process_memory_mb, 2),
                'message': message
            }

        except Exception as e:
            return {
                'status': 'unknown',
                'error': str(e),
                'message': 'Could not check memory'
            }

    def check_config(self) -> Dict[str, Any]:
        """Check critical configuration."""
        issues = []

        # Check encryption key
        encryption_key = os.getenv('ENCRYPTION_KEY')
        if not encryption_key or encryption_key == 'your_encryption_key_here':
            issues.append('ENCRYPTION_KEY not set')

        # Check webhook secret
        webhook_secret = os.getenv('WEBHOOK_SECRET')
        if not webhook_secret or webhook_secret == 'your_secret_key_here':
            issues.append('WEBHOOK_SECRET not set')

        if issues:
            return {
                'status': 'warning',
                'issues': issues,
                'message': f'{len(issues)} configuration issue(s)'
            }
        else:
            return {
                'status': 'healthy',
                'message': 'Critical configuration is set'
            }

    async def check_quick(self) -> Dict[str, Any]:
        """
        Quick health check (database only).

        Use this for frequent health checks (e.g., load balancer).
        """
        db_check = await self.check_database()

        return {
            'status': 'healthy' if db_check['status'] == 'healthy' else 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'database': db_check
        }

    async def check_readiness(self) -> Dict[str, Any]:
        """
        Readiness check for Kubernetes/Docker.

        Returns whether the app is ready to serve traffic.
        """
        # Check database
        db_check = await self.check_database()

        # Check config
        config_check = self.check_config()

        is_ready = (
            db_check['status'] == 'healthy' and
            config_check['status'] in ['healthy', 'warning']
        )

        return {
            'ready': is_ready,
            'timestamp': datetime.now().isoformat(),
            'checks': {
                'database': db_check,
                'config': config_check
            }
        }

    async def check_liveness(self) -> Dict[str, Any]:
        """
        Liveness check for Kubernetes/Docker.

        Returns whether the app is alive (even if not ready).
        """
        return {
            'alive': True,
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }


# Global health check service
_health_check_service: Optional[HealthCheckService] = None


def get_health_check_service() -> HealthCheckService:
    """Get or create global health check service instance."""
    global _health_check_service
    if _health_check_service is None:
        _health_check_service = HealthCheckService()
    return _health_check_service
