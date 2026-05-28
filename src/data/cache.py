"""
M-CACHE: Parquet-backed data cache for OHLCV DataFrames.
Contract: key → parquet file (deterministic, idempotent)
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataCache:
    """Simple parquet-backed cache for OHLCV DataFrames.

    Keys are deterministic strings; values are stored as .parquet files.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ──────────────────────────────────────────────────────

    def get(self, key: str) -> pd.DataFrame | None:
        """Return cached DataFrame or None if not found."""
        path = self._path(key)
        if not path.exists():
            logger.debug("Cache miss: %s", key)
            return None
        try:
            df = pd.read_parquet(path)
            logger.debug("Cache hit: %s (%d rows)", key, len(df))
            return df
        except Exception:
            logger.warning("Corrupt cache file %s, removing", path)
            path.unlink(missing_ok=True)
            return None

    def put(self, key: str, df: pd.DataFrame):
        """Save DataFrame as parquet."""
        path = self._path(key)
        df.to_parquet(path, index=True)
        logger.debug("Cached: %s (%d rows)", key, len(df))

    @staticmethod
    def key(ticker: str, interval: int, start: str, end: str) -> str:
        """Generate deterministic cache key from parameters."""
        return f"{ticker}_{interval}_{start}_{end}"

    # ── internals ──────────────────────────────────────────────────────

    def _path(self, key: str) -> Path:
        safe = key.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe}.parquet"
