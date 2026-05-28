"""
M-BASE: Abstract DataSource interface for financial time-series data.
Contract: N/A (interface contract only)
"""

from abc import ABC, abstractmethod


class DataSource(ABC):
    """Abstract interface for fetching financial time-series data."""

    @abstractmethod
    def fetch_candles(self, ticker: str, interval: int, start: str, end: str) -> "pd.DataFrame":
        """Return OHLCV candles with columns: begin, open, high, low, close, volume, value.

        Args:
            ticker: SECID of the instrument (e.g. 'SBER').
            interval: Candle interval in minutes.
            start: Start date as ISO string YYYY-MM-DD.
            end: End date as ISO string YYYY-MM-DD.

        Returns:
            DataFrame with a DatetimeIndex (column 'begin') and OHLCV columns.
        """
        ...

    @abstractmethod
    def fetch_securities(self) -> "pd.DataFrame":
        """Return metadata for all securities on the configured board.

        Returns:
            DataFrame with columns such as SECID, SHORTNAME, LOTSIZE, etc.
        """
        ...

    @abstractmethod
    def fetch_index_candles(self, index_ticker: str, start: str, end: str) -> "pd.DataFrame":
        """Return OHLCV candles for a market index.

        Args:
            index_ticker: Index identifier (e.g. 'IMOEX').
            start: Start date as ISO string YYYY-MM-DD.
            end: End date as ISO string YYYY-MM-DD.

        Returns:
            DataFrame with a DatetimeIndex and OHLC columns (volume if available).
        """
        ...
