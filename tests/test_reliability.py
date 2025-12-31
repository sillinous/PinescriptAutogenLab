# tests/test_reliability.py

"""
Unit tests for reliability features: retry, reconciliation, backups.
"""

import pytest
import asyncio
import time
import os
import shutil
from datetime import datetime
from backend.reliability.retry_handler import RetryHandler, get_retry_handler
from backend.reliability.reconciliation import OrderReconciliationService, get_reconciliation_service
from backend.reliability.backup_service import BackupService, get_backup_service


@pytest.mark.unit
class TestRetryHandler:
    """Test retry mechanism with exponential backoff."""

    def test_retry_handler_initialization(self):
        """Test retry handler can be initialized."""
        handler = RetryHandler(max_attempts=3, base_delay=0.1)

        assert handler.max_attempts == 3
        assert handler.base_delay == 0.1

    def test_calculate_delay_exponential(self):
        """Test exponential backoff delay calculation."""
        handler = RetryHandler(base_delay=1.0, exponential_base=2)

        delay0 = handler.calculate_delay(0)
        delay1 = handler.calculate_delay(1)
        delay2 = handler.calculate_delay(2)

        # Should follow exponential pattern (with jitter)
        assert 0.8 <= delay0 <= 1.2  # 1.0 ± 20%
        assert 1.6 <= delay1 <= 2.4  # 2.0 ± 20%
        assert 3.2 <= delay2 <= 4.8  # 4.0 ± 20%

    def test_max_delay_cap(self):
        """Test maximum delay is capped."""
        handler = RetryHandler(base_delay=1.0, exponential_base=2, max_delay=5.0)

        delay_large = handler.calculate_delay(10)  # Would be 1024 without cap

        assert delay_large <= 6.0  # 5.0 + max jitter

    def test_successful_operation_no_retry(self):
        """Test successful operation doesn't retry."""
        handler = RetryHandler(max_attempts=3, base_delay=0.01)

        call_count = 0

        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = handler.retry(success_func, operation_type="test")

        assert result == "success"
        assert call_count == 1  # Only called once

    def test_retry_on_failure(self):
        """Test retries on failure."""
        handler = RetryHandler(max_attempts=3, base_delay=0.01)

        call_count = 0

        def fail_twice_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = handler.retry(fail_twice_then_succeed, operation_type="test")

        assert result == "success"
        assert call_count == 3

    def test_exhausted_retries(self):
        """Test exception raised when retries exhausted."""
        handler = RetryHandler(max_attempts=3, base_delay=0.01)

        def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            handler.retry(always_fails, operation_type="test")

    @pytest.mark.asyncio
    async def test_async_retry(self):
        """Test async retry mechanism."""
        handler = RetryHandler(max_attempts=3, base_delay=0.01)

        call_count = 0

        async def async_fail_once():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "async_success"

        result = await handler.retry_async(async_fail_once, operation_type="test")

        assert result == "async_success"
        assert call_count == 2

    def test_retry_with_specific_exceptions(self):
        """Test retry only on specific exceptions."""
        handler = RetryHandler(max_attempts=3, base_delay=0.01, retriable_exceptions=[ConnectionError])

        call_count = 0

        def fail_with_different_exceptions():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Retriable")
            elif call_count == 2:
                raise ValueError("Non-retriable")
            return "success"

        with pytest.raises(ValueError):
            handler.retry(fail_with_different_exceptions, operation_type="test")

        assert call_count == 2

    def test_jitter_variation(self):
        """Test jitter adds randomness to delays."""
        handler = RetryHandler(base_delay=1.0)

        delays = [handler.calculate_delay(1) for _ in range(10)]

        # All delays should be different due to jitter
        assert len(set(delays)) > 1

    def test_singleton_pattern(self):
        """Test retry handler uses singleton pattern."""
        handler1 = get_retry_handler()
        handler2 = get_retry_handler()

        assert handler1 is handler2


@pytest.mark.unit
class TestOrderReconciliation:
    """Test order reconciliation service."""

    def test_reconciliation_service_initialization(self, db):
        """Test reconciliation service can be initialized."""
        service = OrderReconciliationService()

        assert service is not None

    @pytest.mark.asyncio
    async def test_find_stale_orders(self, db):
        """Test finding stale orders."""
        from backend.database import create_order

        # Create an old pending order
        create_order(
            strategy_name='TestStrategy',
            symbol='AAPL',
            action='buy',
            quantity=10,
            order_type='market',
            status='pending',
            broker_order_id='test_order_123'
        )

        # Manually update timestamp to make it old
        from backend.database import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE orders
            SET created_at = datetime('now', '-2 hours')
            WHERE broker_order_id = 'test_order_123'
        """)
        conn.commit()
        conn.close()

        service = OrderReconciliationService()
        stale_orders = await service.find_stale_orders(stale_threshold_minutes=30)

        assert len(stale_orders) >= 1
        assert any(o['broker_order_id'] == 'test_order_123' for o in stale_orders)

    @pytest.mark.asyncio
    async def test_reconcile_order(self, db, mock_alpaca_client):
        """Test reconciling a single order."""
        from backend.database import create_order

        # Create order in database with pending status
        order_id = create_order(
            strategy_name='TestStrategy',
            symbol='AAPL',
            action='buy',
            quantity=10,
            order_type='market',
            status='pending',
            broker_order_id='order_1'
        )

        # Mock broker has it as filled
        mock_alpaca_client.orders['order_1'] = {
            'id': 'order_1',
            'symbol': 'AAPL',
            'status': 'filled',
            'filled_qty': 10,
            'filled_avg_price': 150.0
        }

        service = OrderReconciliationService()

        # Get order from database
        from backend.database import get_order
        order = get_order(order_id)

        # Reconcile
        result = await service.reconcile_order(order)

        assert result['reconciled'] is True
        assert result['status_changed'] is True

        # Verify database updated
        updated_order = get_order(order_id)
        assert updated_order['status'] == 'filled'

    @pytest.mark.asyncio
    async def test_reconciliation_no_changes_needed(self, db, mock_alpaca_client):
        """Test reconciliation when statuses match."""
        from backend.database import create_order

        # Create order with filled status
        order_id = create_order(
            strategy_name='TestStrategy',
            symbol='AAPL',
            action='buy',
            quantity=10,
            order_type='market',
            status='filled',
            broker_order_id='order_1'
        )

        # Mock broker also has it as filled
        mock_alpaca_client.orders['order_1'] = {
            'id': 'order_1',
            'symbol': 'AAPL',
            'status': 'filled',
            'filled_qty': 10,
            'filled_avg_price': 150.0
        }

        service = OrderReconciliationService()

        from backend.database import get_order
        order = get_order(order_id)

        result = await service.reconcile_order(order)

        assert result['reconciled'] is True
        assert result['status_changed'] is False

    @pytest.mark.asyncio
    async def test_reconcile_order_not_found_on_broker(self, db, mock_alpaca_client):
        """Test reconciling an order that is not found on the broker."""
        from backend.database import create_order, get_order

        # Create order in database with pending status
        order_id = create_order(
            strategy_name='TestStrategy',
            symbol='AAPL',
            action='buy',
            quantity=10,
            order_type='market',
            status='pending',
            broker_order_id='order_not_found'
        )

        service = OrderReconciliationService()
        order = get_order(order_id)

        # Reconcile
        result = await service.reconcile_order(order)

        assert result['reconciled'] is True
        assert result['status_changed'] is True

        # Verify database updated
        updated_order = get_order(order_id)
        assert updated_order['status'] == 'not_found'

    def test_singleton_pattern(self):
        """Test reconciliation service uses singleton pattern."""
        service1 = get_reconciliation_service()
        service2 = get_reconciliation_service()

        assert service1 is service2


@pytest.mark.unit
class TestBackupService:
    """Test backup and restore service."""

    def test_backup_service_initialization(self, db):
        """Test backup service can be initialized."""
        service = BackupService()

        assert service is not None
        assert os.path.exists(service.backup_dir)

    def test_create_backup(self, db, cleanup_temp_files):
        """Test creating a database backup."""
        service = BackupService()

        result = service.create_backup(
            compress=False,
            encrypt=False,
            description="Test backup"
        )

        assert result['success'] is True
        assert 'backup_name' in result
        assert 'backup_path' in result
        assert os.path.exists(result['backup_path'])

        # Add to cleanup
        cleanup_temp_files.append(result['backup_path'])

    def test_create_compressed_backup(self, db, cleanup_temp_files):
        """Test creating compressed backup."""
        service = BackupService()

        result = service.create_backup(
            compress=True,
            encrypt=False,
            description="Compressed backup"
        )

        assert result['success'] is True
        assert result['backup_path'].endswith('.gz')
        assert os.path.exists(result['backup_path'])

        # Compressed should be smaller than original
        # (though with small test DB, this might not always be true)

        cleanup_temp_files.append(result['backup_path'])

    def test_create_encrypted_backup(self, db, encryption_key, cleanup_temp_files):
        """Test creating encrypted backup."""
        os.environ['ENCRYPTION_KEY'] = encryption_key

        service = BackupService()

        result = service.create_backup(
            compress=False,
            encrypt=True,
            description="Encrypted backup"
        )

        assert result['success'] is True
        assert result['encrypted'] is True
        assert os.path.exists(result['backup_path'])

        cleanup_temp_files.append(result['backup_path'])

    def test_list_backups(self, db, cleanup_temp_files):
        """Test listing backups."""
        service = BackupService()

        # Create multiple backups
        result1 = service.create_backup(compress=False, encrypt=False, description="Backup 1")
        result2 = service.create_backup(compress=False, encrypt=False, description="Backup 2")

        cleanup_temp_files.extend([result1['backup_path'], result2['backup_path']])

        # List backups
        backups = service.list_backups()

        assert len(backups) >= 2
        assert any(b['name'] == result1['backup_name'] for b in backups)
        assert any(b['name'] == result2['backup_name'] for b in backups)

    def test_restore_backup(self, db, cleanup_temp_files):
        """Test restoring from backup."""
        from backend.database import create_user, get_user_by_username

        # Create test data
        user_id = create_user(
            username='backuptest',
            email='backup@test.com',
            password_hash='hash123',
            full_name='Backup Test'
        )

        service = BackupService()

        # Create backup
        backup_result = service.create_backup(compress=False, encrypt=False, description="Test restore")
        cleanup_temp_files.append(backup_result['backup_path'])

        # Modify database
        from backend.database import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE username = 'backuptest'")
        conn.commit()
        conn.close()

        # Verify user is gone
        user = get_user_by_username('backuptest')
        assert user is None

        # Restore backup
        restore_result = service.restore_backup(backup_result['backup_name'])

        assert restore_result['success'] is True

        # Verify user is back
        user = get_user_by_username('backuptest')
        assert user is not None
        assert user['username'] == 'backuptest'

    def test_verify_backup(self, db, cleanup_temp_files):
        """Test verifying backup integrity."""
        service = BackupService()

        # Create backup
        result = service.create_backup(compress=False, encrypt=False, description="Verification test")
        cleanup_temp_files.append(result['backup_path'])

        # Verify backup
        verify_result = service.verify_backup(result['backup_name'])

        assert verify_result['valid'] is True
        assert verify_result['can_open'] is True
        assert verify_result['tables_count'] > 0

    def test_cleanup_old_backups(self, db, cleanup_temp_files):
        """Test cleaning up old backups."""
        service = BackupService()

        # Create backup
        result = service.create_backup(compress=False, encrypt=False, description="Old backup")
        cleanup_temp_files.append(result['backup_path'])

        # Manually set old modification time
        old_time = time.time() - (100 * 24 * 60 * 60)  # 100 days ago
        os.utime(result['backup_path'], (old_time, old_time))

        # Cleanup backups older than 30 days
        cleanup_result = service.cleanup_old_backups(days_old=30)

        assert cleanup_result['deleted_count'] >= 1
        assert not os.path.exists(result['backup_path'])

    def test_automatic_backup_retention(self, db, cleanup_temp_files):
        """Test automatic backup retention policy."""
        service = BackupService()
        service.retention_days = 7

        # Create multiple backups
        backups = []
        for i in range(5):
            result = service.create_backup(compress=False, encrypt=False, description=f"Backup {i}")
            backups.append(result)
            cleanup_temp_files.append(result['backup_path'])

        # Verify all created
        assert len(service.list_backups()) >= 5

    def test_backup_metadata(self, db, cleanup_temp_files):
        """Test backup includes metadata."""
        service = BackupService()

        description = "Backup with metadata"
        result = service.create_backup(
            compress=False,
            encrypt=False,
            description=description
        )

        cleanup_temp_files.append(result['backup_path'])

        # Check metadata
        backups = service.list_backups()
        backup = next(b for b in backups if b['name'] == result['backup_name'])

        assert backup['description'] == description
        assert 'created_at' in backup
        assert 'size_bytes' in backup

    def test_restore_corrupted_backup(self, db, cleanup_temp_files):
        """Test restoring from a corrupted backup."""
        service = BackupService()

        # Create a valid backup
        backup_result = service.create_backup(compress=False, encrypt=False, description="Corrupted backup test")
        cleanup_temp_files.append(backup_result['backup_path'])

        # Corrupt the backup file
        with open(backup_result['backup_path'], 'w') as f:
            f.write("This is not a valid database file.")

        # Attempt to restore
        restore_result = service.restore_backup(backup_result['backup_name'])

        assert restore_result['success'] is False
        assert 'error' in restore_result

    def test_singleton_pattern(self):
        """Test backup service uses singleton pattern."""
        service1 = get_backup_service()
        service2 = get_backup_service()

        assert service1 is service2
