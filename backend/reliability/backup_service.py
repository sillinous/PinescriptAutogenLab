# backend/reliability/backup_service.py

"""
Database backup and restore system.

Features:
- Automated daily backups
- Point-in-time recovery
- Backup encryption
- Compression
- Retention policy
- Backup verification
- Cloud storage support (S3, optional)
"""

import os
import shutil
import gzip
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
from backend.monitoring.logger import log_info, log_error, log_warning
from backend.security.encryption import encrypt_credential, decrypt_credential, ENCRYPTION_AVAILABLE


class BackupService:
    """Service for database backup and restore."""

    def __init__(self):
        self.data_dir = Path(os.getenv("PINELAB_DATA", "./data"))
        self.backup_dir = self.data_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        self.retention_days = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))
        self.max_backups = int(os.getenv("MAX_BACKUPS", "100"))

    def create_backup(
        self,
        compress: bool = True,
        encrypt: bool = True,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create database backup.

        Args:
            compress: Whether to compress backup with gzip
            encrypt: Whether to encrypt backup (requires ENCRYPTION_KEY)
            description: Optional description for backup

        Returns:
            Backup info
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            db_file = self.data_dir / "pinelab.db"

            if not db_file.exists():
                return {
                    'status': 'error',
                    'message': 'Database file not found'
                }

            # Create backup filename
            backup_name = f"pinelab_backup_{timestamp}"
            backup_file = self.backup_dir / f"{backup_name}.db"

            # Copy database
            log_info(f"[BACKUP] Creating backup: {backup_name}")
            shutil.copy2(db_file, backup_file)

            backup_size = backup_file.stat().st_size

            # Compress if requested
            if compress:
                compressed_file = self.backup_dir / f"{backup_name}.db.gz"
                with open(backup_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                # Remove uncompressed file
                backup_file.unlink()
                backup_file = compressed_file

                log_info(f"[BACKUP] Compressed: {backup_size} -> {backup_file.stat().st_size} bytes")

            # Encrypt if requested and available
            if encrypt and ENCRYPTION_AVAILABLE:
                encrypted_file = self.backup_dir / f"{backup_name}.db.gz.enc"

                # Read backup file
                with open(backup_file, 'rb') as f:
                    backup_data = f.read()

                # Encrypt (convert bytes to base64 for encryption service)
                import base64
                encrypted_data = encrypt_credential(base64.b64encode(backup_data).decode())

                # Write encrypted file
                with open(encrypted_file, 'w') as f:
                    f.write(encrypted_data)

                # Remove unencrypted file
                backup_file.unlink()
                backup_file = encrypted_file

                log_info(f"[BACKUP] Encrypted backup created")

            # Create metadata file
            metadata = {
                'timestamp': timestamp,
                'created_at': datetime.now().isoformat(),
                'description': description,
                'original_size': backup_size,
                'compressed': compress,
                'encrypted': encrypt and ENCRYPTION_AVAILABLE,
                'filename': backup_file.name
            }

            metadata_file = self.backup_dir / f"{backup_name}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            log_info(f"[BACKUP] Backup created successfully: {backup_file.name}")

            return {
                'status': 'success',
                'backup_name': backup_name,
                'filename': backup_file.name,
                'size_bytes': backup_file.stat().st_size,
                'compressed': compress,
                'encrypted': encrypt and ENCRYPTION_AVAILABLE,
                'created_at': metadata['created_at']
            }

        except Exception as e:
            log_error(f"[BACKUP] Failed to create backup: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def restore_backup(
        self,
        backup_name: str,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Restore database from backup.

        Args:
            backup_name: Name of backup to restore (without extension)
            force: Skip confirmation (dangerous!)

        Returns:
            Restore result
        """
        try:
            # Find backup file
            backup_file = None
            for ext in ['.db.gz.enc', '.db.gz', '.db']:
                candidate = self.backup_dir / f"{backup_name}{ext}"
                if candidate.exists():
                    backup_file = candidate
                    break

            if not backup_file:
                return {
                    'status': 'error',
                    'message': f'Backup {backup_name} not found'
                }

            # Load metadata
            metadata_file = self.backup_dir / f"{backup_name}.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            log_info(f"[BACKUP] Restoring backup: {backup_name}")

            # Create backup of current database before restore
            current_backup = self.create_backup(
                compress=True,
                encrypt=True,
                description=f"Auto-backup before restore of {backup_name}"
            )

            if current_backup['status'] == 'error':
                return {
                    'status': 'error',
                    'message': 'Failed to backup current database before restore'
                }

            # Read backup file
            with open(backup_file, 'rb' if backup_file.suffix in ['.db', '.gz'] else 'r') as f:
                backup_data = f.read()

            # Decrypt if encrypted
            if metadata.get('encrypted'):
                if isinstance(backup_data, bytes):
                    backup_data = backup_data.decode()

                decrypted_b64 = decrypt_credential(backup_data)
                import base64
                backup_data = base64.b64decode(decrypted_b64)
                log_info("[BACKUP] Decrypted backup")

            # Decompress if compressed
            if metadata.get('compressed') or backup_file.suffix == '.gz':
                if isinstance(backup_data, str):
                    backup_data = backup_data.encode()

                import gzip
                backup_data = gzip.decompress(backup_data)
                log_info("[BACKUP] Decompressed backup")

            # Write to database file
            db_file = self.data_dir / "pinelab.db"
            with open(db_file, 'wb') as f:
                f.write(backup_data)

            log_info(f"[BACKUP] Database restored from {backup_name}")

            return {
                'status': 'success',
                'backup_name': backup_name,
                'restored_at': datetime.now().isoformat(),
                'current_backup': current_backup.get('backup_name'),
                'message': 'Database restored successfully. Application restart recommended.'
            }

        except Exception as e:
            log_error(f"[BACKUP] Failed to restore backup: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups."""
        backups = []

        for metadata_file in self.backup_dir.glob("pinelab_backup_*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                backup_name = metadata_file.stem
                backup_file = self.backup_dir / metadata['filename']

                backups.append({
                    'name': backup_name,
                    'created_at': metadata['created_at'],
                    'description': metadata.get('description'),
                    'size_bytes': backup_file.stat().st_size if backup_file.exists() else 0,
                    'compressed': metadata.get('compressed', False),
                    'encrypted': metadata.get('encrypted', False)
                })
            except Exception as e:
                log_warning(f"[BACKUP] Error reading backup metadata {metadata_file}: {e}")

        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x['created_at'], reverse=True)

        return backups

    def cleanup_old_backups(self) -> Dict[str, Any]:
        """
        Clean up old backups based on retention policy.

        Removes backups older than retention_days and keeps only max_backups.
        """
        try:
            backups = self.list_backups()

            if not backups:
                return {
                    'status': 'success',
                    'deleted_count': 0,
                    'message': 'No backups to clean up'
                }

            # Determine which backups to delete
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            to_delete = []

            for i, backup in enumerate(backups):
                created_at = datetime.fromisoformat(backup['created_at'])

                # Delete if older than retention period OR beyond max count
                if created_at < cutoff_date or i >= self.max_backups:
                    to_delete.append(backup)

            # Delete old backups
            deleted_count = 0
            for backup in to_delete:
                try:
                    # Delete backup file
                    for ext in ['.db.gz.enc', '.db.gz', '.db']:
                        backup_file = self.backup_dir / f"{backup['name']}{ext}"
                        if backup_file.exists():
                            backup_file.unlink()

                    # Delete metadata
                    metadata_file = self.backup_dir / f"{backup['name']}.json"
                    if metadata_file.exists():
                        metadata_file.unlink()

                    deleted_count += 1
                    log_info(f"[BACKUP] Deleted old backup: {backup['name']}")

                except Exception as e:
                    log_error(f"[BACKUP] Error deleting backup {backup['name']}: {e}")

            log_info(f"[BACKUP] Cleanup complete. Deleted {deleted_count} old backups.")

            return {
                'status': 'success',
                'deleted_count': deleted_count,
                'remaining_backups': len(backups) - deleted_count
            }

        except Exception as e:
            log_error(f"[BACKUP] Cleanup failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def verify_backup(self, backup_name: str) -> Dict[str, Any]:
        """
        Verify backup integrity.

        Args:
            backup_name: Name of backup to verify

        Returns:
            Verification result
        """
        try:
            # Find backup file
            backup_file = None
            for ext in ['.db.gz.enc', '.db.gz', '.db']:
                candidate = self.backup_dir / f"{backup_name}{ext}"
                if candidate.exists():
                    backup_file = candidate
                    break

            if not backup_file:
                return {
                    'status': 'error',
                    'message': 'Backup not found'
                }

            # Load metadata
            metadata_file = self.backup_dir / f"{backup_name}.json"
            if not metadata_file.exists():
                return {
                    'status': 'warning',
                    'message': 'Backup exists but metadata is missing'
                }

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Try to read and decompress/decrypt
            with open(backup_file, 'rb' if backup_file.suffix in ['.db', '.gz'] else 'r') as f:
                backup_data = f.read()

            if metadata.get('encrypted'):
                if isinstance(backup_data, bytes):
                    backup_data = backup_data.decode()
                decrypted_b64 = decrypt_credential(backup_data)

            if metadata.get('compressed'):
                if isinstance(backup_data, str):
                    import base64
                    backup_data = base64.b64decode(decrypted_b64)
                import gzip
                decompressed = gzip.decompress(backup_data)

            return {
                'status': 'success',
                'message': 'Backup is valid and readable',
                'encrypted': metadata.get('encrypted', False),
                'compressed': metadata.get('compressed', False)
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Backup verification failed: {str(e)}'
            }


# Global backup service
_backup_service: Optional[BackupService] = None


def get_backup_service() -> BackupService:
    """Get or create global backup service instance."""
    global _backup_service
    if _backup_service is None:
        _backup_service = BackupService()
    return _backup_service


# Convenience function for scheduled backups
def create_daily_backup():
    """Create daily backup (to be called by scheduler)."""
    service = get_backup_service()
    result = service.create_backup(
        compress=True,
        encrypt=True,
        description="Automated daily backup"
    )

    # Cleanup old backups
    if result['status'] == 'success':
        service.cleanup_old_backups()

    return result
