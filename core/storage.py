"""
core/storage.py — Per-user file storage isolation.

Each user gets their own sandbox directory: storage/{user_id}/
This prevents cross-user data leakage and enables independent cleanup.

For production: swap LocalStorage for S3Storage / GCSStorage by
implementing the same interface.
"""
import os
import shutil
import hashlib
import time
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_STORAGE = os.getenv("STORAGE_BASE", "storage")
MAX_USER_STORAGE_MB = int(os.getenv("MAX_USER_STORAGE_MB", "500"))
FILE_TTL_HOURS = int(os.getenv("FILE_TTL_HOURS", "24"))


class UserStorage:
    """
    Manages all file I/O for a single user session.
    Thread-safe for async use (no shared mutable state between instances).
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.root = Path(BASE_STORAGE) / user_id
        self.sandbox = self.root / "sandbox"
        self.models = self.root / "models"
        self.reports = self.root / "reports"
        self.charts = self.root / "charts"
        self._ensure_dirs()

    def _ensure_dirs(self):
        for d in [self.sandbox, self.models, self.reports, self.charts]:
            d.mkdir(parents=True, exist_ok=True)

    # ── Paths ─────────────────────────────────────────────────────────

    def sandbox_path(self, filename: str) -> str:
        return str(self.sandbox / filename)

    def model_path(self, model_name: str) -> str:
        return str(self.models / model_name)

    def report_path(self, name: str) -> str:
        return str(self.reports / name)

    def chart_path(self, name: str) -> str:
        return str(self.charts / name)

    # ── File operations ───────────────────────────────────────────────

    def save_upload(self, raw_bytes: bytes, original_filename: str) -> str:
        """
        Save an uploaded file. Returns the local path.
        Deduplicates by SHA-256: identical files share the same path.
        """
        file_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]
        dest = self.sandbox / f"{original_filename}"
        dest.write_bytes(raw_bytes)
        return str(dest)

    def list_pickles(self) -> Dict[str, str]:
        """Returns {filename: path} for all .pkl files in sandbox."""
        return {
            f.name: str(f)
            for f in self.sandbox.glob("*.pkl")
        }

    def list_uploads(self) -> Dict[str, str]:
        """Returns {filename: path} for all uploaded raw files."""
        result = {}
        for ext in ["*.csv", "*.xlsx", "*.xls", "*.parquet", "*.json"]:
            for f in self.sandbox.glob(ext):
                result[f.name] = str(f)
        return result

    def storage_mb(self) -> float:
        total = sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())
        return round(total / 1_048_576, 2)

    def check_quota(self):
        used = self.storage_mb()
        if used > MAX_USER_STORAGE_MB:
            raise QuotaExceededError(
                f"Storage quota exceeded: {used}MB used / {MAX_USER_STORAGE_MB}MB limit. "
                "Download and delete files to continue."
            )

    def cleanup(self):
        """Remove entire user storage directory."""
        if self.root.exists():
            shutil.rmtree(self.root)
            logger.info(f"Cleaned up storage for user {self.user_id}")

    def cleanup_old_files(self, ttl_hours: int = FILE_TTL_HOURS):
        """Remove files older than ttl_hours from sandbox (not models/reports)."""
        cutoff = time.time() - ttl_hours * 3600
        removed = 0
        for f in self.sandbox.glob("*"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        if removed:
            logger.info(f"Removed {removed} stale files for user {self.user_id}")


class QuotaExceededError(Exception):
    pass


# ── Global cleanup job (call from FastAPI lifespan) ───────────────────

def cleanup_stale_users(ttl_hours: int = FILE_TTL_HOURS):
    """
    Remove user directories where all files are older than ttl_hours.
    Run periodically (e.g. every hour via background task).
    """
    base = Path(BASE_STORAGE)
    if not base.exists():
        return
    cutoff = time.time() - ttl_hours * 3600
    for user_dir in base.iterdir():
        if not user_dir.is_dir():
            continue
        files = list(user_dir.rglob("*"))
        if files and all(f.stat().st_mtime < cutoff for f in files if f.is_file()):
            shutil.rmtree(user_dir)
            logger.info(f"Evicted stale user directory: {user_dir.name}")


def get_storage(user_id: str) -> UserStorage:
    return UserStorage(user_id)
