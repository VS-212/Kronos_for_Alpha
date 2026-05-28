"""
M-FETCH: MOEX ISS data fetcher — download 10-min OHLCV candles
Contract: [start, end, interval, tickers] → [parquet files, manifest.json]
Status: ✅ ready

Usage:
    from src.data.fetcher import fetch_and_save
    data = fetch_and_save("2023-01-01", "2026-05-01")

CLI:
    python -m src.data.fetcher --start 2023-01-01 --end 2026-05-01

Output: data/v3/raw/{TICKER}.parquet
        data/v3/raw/manifest.json

Known failures:
  - "requests.exceptions.HTTPError: 429 Client Error"
    → Rate-limited. Built-in exponential backoff 2^n*2s handles this.
      If persistent: reduce ThreadPoolExecutor workers from 5 to 2.

  - "RuntimeError: Failed to fetch {ticker}"
    → 5 retries exhausted. Check internet, MOEX ISS status.
      Resume: re-run with --resume (default), will skip completed chunks.

  - "requests.exceptions.ConnectionError: HTTPSConnectionPool"
    → Network down or MOEX ISS unreachable. Built-in retry 2^n*5s.
      If persistent: wait 5 min, re-run with --resume.

See docs/operations/failures.md for failure catalog.
See docs/conventions/cli.md for CLI standard.
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import apimoex
import pandas as pd
import requests
from requests.exceptions import RequestException

from src.data.base import DataSource

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────

TICKERS = [
    "SBER", "GAZP", "LKOH", "ALRS", "ROSN", "NVTK", "PLZL", "GMKN",
    "TATN", "VTBR", "CHMF", "NLMK", "MAGN", "AFLT", "FIVE", "MOEX",
    "TCST", "YNDX", "SNGS", "SNGSP",
]
"""20 liquid MOEX equities (top by volume, all sectors)."""

MACRO_TICKERS = ["IMOEX"]
"""Moscow Exchange Index — separate endpoint, used as macro context."""

LOT_SIZES = {
    "SBER": 10, "GAZP": 10, "LKOH": 1, "ALRS": 10, "ROSN": 10,
    "NVTK": 10, "PLZL": 1, "GMKN": 1, "TATN": 1, "VTBR": 1000,
    "CHMF": 1, "NLMK": 10, "MAGN": 10, "AFLT": 10, "FIVE": 1,
    "MOEX": 10, "TCST": 1, "YNDX": 1, "SNGS": 100, "SNGSP": 100,
}
"""MOEX lot size (paper share → whole share)."""

URL_STOCK = (
    "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities"
    "/{ticker}/candles.json"
)
"""TQBR equities endpoint."""

URL_INDEX = (
    "https://iss.moex.com/iss/engines/stock/markets/index/securities"
    "/{ticker}/candles.json"
)
"""Index candles endpoint."""

URL_INDEX_HISTORY = (
    "https://iss.moex.com/iss/history/engines/stock/"
    "markets/index/securities/{ticker}.json"
)
"""Index daily history endpoint (cursor-paginated)."""

DELAY: float = 0.5
"""Delay between HTTP requests (seconds). MOEX ISS tolerates ~1-2 req/s."""

MAX_RETRIES: int = 5
"""Retry attempts on network/HTTP error."""

PAGE_SIZE: int = 500
"""Max records per page (API limit)."""

REQUEST_TIMEOUT: int = 60
"""HTTP request timeout (seconds)."""

BARS_PER_TRADING_DAY: int = 52
"""10-min bars in main session (10:00–18:40)."""

_CANDLE_COLUMNS_MAP = {
    "begin": "begin",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "value": "value",
    "volume": "volume",
}

_INDEX_COLUMNS_MAP = {
    "TRADEDATE": "begin",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "CLOSE": "close",
    "VOLUME": "volume",
}


# ── Utilities ──────────────────────────────────────────────────────────────

def _is_macro(ticker: str) -> bool:
    return ticker in MACRO_TICKERS


def _expected_bars(start_date: str, end_date: str) -> int:
    """Approximate bar count in range (52 bars × trading days)."""
    s = datetime.strptime(start_date, "%Y-%m-%d")
    e = datetime.strptime(end_date, "%Y-%m-%d")
    trading_days = 0
    cur = s
    while cur <= e:
        if cur.weekday() < 5:
            trading_days += 1
        cur += timedelta(days=1)
    return trading_days * BARS_PER_TRADING_DAY


def _month_chunks(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Split range into monthly intervals (MOEX dislikes large requests)."""
    s = datetime.strptime(start_date, "%Y-%m-%d")
    e = datetime.strptime(end_date, "%Y-%m-%d")
    chunks = []
    cur = s
    while cur <= e:
        month_end = cur.replace(day=28) + timedelta(days=4)
        month_end = month_end.replace(day=1) - timedelta(days=1)
        chunk_end = min(month_end, e)
        chunks.append((cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = chunk_end + timedelta(days=1)
    return chunks


# ── DataSource implementation ──────────────────────────────────────────────

class Fetcher(DataSource):
    """MOEX ISS API data fetcher implementing the DataSource interface.

    Fetches OHLCV candles via apimoex for stocks and the index candles
    endpoint for indices.  Provides 5→1 minute fallback with resampling
    when interval=5 returns nothing.  All API calls use exponential
    backoff and rate limiting.

    Parameters
    ----------
    board : str
        MOEX board name (default ``"TQBR"``).
    market : str
        Market segment (default ``"shares"``).
    engine : str
        Trading engine (default ``"stock"``).
    max_retries : int
        Number of retries on transient HTTP / ISS errors.
    retry_delay : float
        Base seconds between retries.
    rate_limit_delay : float
        Minimum seconds between consecutive API calls.
    timezone : str
        Timezone for index localization (default ``"Europe/Moscow"``).
    """

    def __init__(
        self,
        board: str = "TQBR",
        market: str = "shares",
        engine: str = "stock",
        max_retries: int = 5,
        retry_delay: float = 2.0,
        rate_limit_delay: float = 0.5,
        timezone: str = "Europe/Moscow",
    ):
        self.board = board
        self.market = market
        self.engine = engine
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_delay = rate_limit_delay
        self.timezone = timezone
        self._last_request_time: float = 0.0

    # ── DataSource interface ──────────────────────────────────────────

    def fetch_candles(
        self, ticker: str, interval: int, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch OHLCV candles for a single ticker.

        Uses apimoex for stock tickers; raw ISS candles endpoint for
        index tickers.  Tries the requested interval first.  For
        interval=5, falls back to interval=1 with 5-min resampling when
        the primary request returns nothing.

        Returns
        -------
        pd.DataFrame
            Columns: ``[begin, open, high, low, close, volume, value]``.
            Indexed by ``begin`` as a DatetimeIndex (MSK).
        """
        logger.info(
            "Fetching candles: %s interval=%d %s → %s", ticker, interval, start, end
        )

        if _is_macro(ticker):
            primary = self._fetch_index_candles_intraday(ticker, interval, start, end)
        else:
            primary = self._fetch_candles_raw(ticker, interval, start, end)

        if primary is not None and not primary.empty:
            return primary

        if interval == 5:
            logger.info(
                "Fallback to 1-min candles for %s (%s → %s) resampling to 5-min",
                ticker, start, end,
            )
            if _is_macro(ticker):
                df_1m = self._fetch_index_candles_intraday(ticker, 1, start, end)
            else:
                df_1m = self._fetch_candles_raw(ticker, 1, start, end)

            if df_1m is None or df_1m.empty:
                logger.warning("Fallback also empty for %s", ticker)
                return pd.DataFrame()
            df_5 = self._resample_to_5min(df_1m)
            df_5 = df_5.dropna(subset=["close"])
            self._localize_index(df_5)
            return df_5

        logger.warning("No data returned for %s interval=%d", ticker, interval)
        return pd.DataFrame()

    def fetch_securities(self) -> pd.DataFrame:
        """Return metadata for all securities on the configured board.

        Merges the static metadata table with the live marketdata table.

        Returns
        -------
        pd.DataFrame
            Indexed by SECID.  Columns include SHORTNAME, LOTSIZE, LAST, etc.
        """
        logger.info("Fetching securities metadata for board=%s", self.board)

        meta = self._get_with_retry(
            lambda sess: apimoex.get_board_securities(sess, board=self.board)
        )
        if meta is None:
            return pd.DataFrame()

        marketdata = self._get_with_retry(
            lambda sess: apimoex.get_board_securities(
                sess, table="marketdata", board=self.board
            )
        )

        df_meta = pd.DataFrame(meta)
        if df_meta.empty:
            return df_meta

        if marketdata and "SECID" in (md_df := pd.DataFrame(marketdata)).columns:
            df_meta = df_meta.merge(md_df, on="SECID", how="left", suffixes=("", "_md"))

        df_meta.set_index("SECID", inplace=True)
        return df_meta

    def fetch_index_candles(
        self, index_ticker: str, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch daily OHLCV history for a MOEX index (e.g. IMOEX).

        Uses ``ISSClient`` with cursor-based pagination for the ISS
        history endpoint.

        Returns
        -------
        pd.DataFrame
            Columns: ``[begin, open, high, low, close, volume]``.
            Indexed by ``begin`` (datetime, MSK timezone).
        """
        logger.info("Fetching index candles: %s %s → %s", index_ticker, start, end)

        url = (
            f"https://iss.moex.com/iss/history/engines/{self.engine}/"
            f"markets/index/securities/{index_ticker}.json"
        )
        query = {
            "iss.only": "history,history.cursor",
            "history.columns": "TRADEDATE,CLOSE,OPEN,HIGH,LOW,VOLUME",
            "from": start,
            "till": end,
        }

        raw = self._get_with_retry(
            lambda sess, u=url, q=query: _iss_get_all(sess, u, q)
        )

        if raw is None:
            logger.warning("No index data for %s", index_ticker)
            return pd.DataFrame()

        table = raw.get("history")
        if table is None or not table:
            logger.warning("Index history empty for %s", index_ticker)
            return pd.DataFrame()

        df = pd.DataFrame(table)
        df = df.rename(columns=_INDEX_COLUMNS_MAP)
        df = df[[c for c in _INDEX_COLUMNS_MAP.values() if c in df.columns]]

        if "begin" in df.columns:
            df["begin"] = pd.to_datetime(df["begin"])
            df.set_index("begin", inplace=True)

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.sort_index(inplace=True)
        df = df.dropna(subset=["close"])
        self._localize_index(df)
        return df

    # ── Internal: candle fetching ─────────────────────────────────────

    def _fetch_candles_raw(
        self, ticker: str, interval: int, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Single-shot stock candle fetch via apimoex. Returns DataFrame or None."""
        raw = self._get_with_retry(
            lambda sess: apimoex.get_board_candles(
                sess,
                ticker,
                interval=interval,
                board=self.board,
                start=start,
                end=end,
            )
        )

        if raw is None:
            return None
        if isinstance(raw, list) and len(raw) == 0:
            return pd.DataFrame()

        if not isinstance(raw, list):
            logger.warning(
                "Unexpected API response type for %s: %s", ticker, type(raw)
            )
            return None

        df = pd.DataFrame(raw)

        required = {"begin", "open", "close"}
        if not required.issubset(df.columns):
            logger.warning(
                "Candle response missing required columns for %s. Got: %s",
                ticker, list(df.columns),
            )
            return None

        df = df.rename(columns=_CANDLE_COLUMNS_MAP)
        keep = [c for c in _CANDLE_COLUMNS_MAP.values() if c in df.columns]
        df = df[keep]

        df["begin"] = pd.to_datetime(df["begin"])
        df.set_index("begin", inplace=True)

        numeric_cols = [
            c for c in ["open", "high", "low", "close", "volume", "value"]
            if c in df.columns
        ]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        df.sort_index(inplace=True)
        self._localize_index(df)
        return df

    def _fetch_index_candles_intraday(
        self, ticker: str, interval: int, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch intraday candles for an index ticker via the ISS candles endpoint.

        Uses offset-based pagination (same as fetcher_v3).  Does NOT use
        apimoex because index tickers are not on a board.
        """
        logger.info(
            "Fetching index intraday: %s interval=%d %s → %s",
            ticker, interval, start, end,
        )

        expected = _expected_bars(start, end)
        all_frames: list[pd.DataFrame] = []
        offset = 0

        while True:
            url = URL_INDEX.format(ticker=ticker)
            query = {
                "from": start,
                "till": end,
                "interval": str(interval),
                "start": str(offset),
                "limit": str(PAGE_SIZE),
                "iss.meta": "off",
                "candles.columns": "begin,open,high,low,close,volume",
            }
            params = "&".join(f"{k}={v}" for k, v in query.items())
            full_url = f"{url}?{params}"

            raw = self._get_with_retry(
                lambda sess, u=full_url: _fetch_json_page(sess, u)
            )
            if raw is None:
                break

            df = self._parse_index_candle_page(raw)
            if df is None or len(df) == 0:
                break

            all_frames.append(df)
            offset += PAGE_SIZE
            self._rate_limit()

        if not all_frames:
            return pd.DataFrame()

        result = pd.concat(all_frames, ignore_index=True)
        result = result.drop_duplicates(subset=["begin"]).sort_values("begin")
        result.set_index("begin", inplace=True)

        got = len(result)
        if got < expected * 0.5:
            logger.info(
                "%s: got %d bars (expected ~%d)", ticker, got, expected
            )

        self._localize_index(result)
        return result

    @staticmethod
    def _parse_index_candle_page(data: dict) -> Optional[pd.DataFrame]:
        """Parse ISS candles JSON block → DataFrame with 'begin' column."""
        try:
            candles = data.get("candles", {})
            cols = candles.get("columns", [])
            rows = candles.get("data", [])
            if not rows:
                return None
            df = pd.DataFrame(rows, columns=cols)
            df["begin"] = pd.to_datetime(df["begin"])
            return df
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Index candle parse error: %s", e)
            return None

    # ── Internal: HTTP layer ──────────────────────────────────────────

    def _rate_limit(self):
        """Ensure minimum delay between API calls."""
        if self.rate_limit_delay <= 0:
            return
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)

    def _get_with_retry(self, func):
        """Call *func* with a fresh requests.Session, retrying with
        exponential backoff on failure.

        ``func`` receives a ``requests.Session`` as its sole argument
        (wrapped in a context manager).  On 429 responses the backoff
        multiplier doubles each retry.
        """
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            self._rate_limit()
            try:
                with requests.Session() as session:
                    result = func(session)
                self._last_request_time = time.time()
                return result
            except RequestException as exc:
                last_exc = exc
                status = None
                if hasattr(exc, "response") and exc.response is not None:
                    status = exc.response.status_code
                wait = (
                    self.retry_delay * (2 ** (attempt - 1))
                    if status == 429
                    else self.retry_delay
                )
                logger.warning(
                    "HTTP error on attempt %d/%d (status=%s, wait=%.1fs): %s",
                    attempt, self.max_retries, status, wait, exc,
                )
            except Exception as exc:
                last_exc = exc
                logger.error(
                    "Unexpected error on attempt %d/%d: %s",
                    attempt, self.max_retries, exc,
                )
            if attempt < self.max_retries:
                wait = self.retry_delay * (2 ** (attempt - 1))
                time.sleep(wait)
        logger.error(
            "All %d retries exhausted. Last error: %s", self.max_retries, last_exc
        )
        return None

    # ── Internal: post-processing ─────────────────────────────────────

    def _localize_index(self, df: pd.DataFrame):
        """Convert naive UTC index to timezone-aware MSK."""
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return
        if df.index.tz is not None:
            return
        df.index = df.index.tz_localize("UTC").tz_convert(self.timezone)

    @staticmethod
    def _resample_to_5min(df_1m: pd.DataFrame) -> pd.DataFrame:
        """Resample a 1-minute DataFrame to 5-minute OHLCV bars."""
        ohlc = df_1m["close"].resample("5min").ohlc()
        ohlc.columns = ["open", "high", "low", "close"]

        df_5 = pd.DataFrame(index=ohlc.index)
        df_5["open"] = ohlc["open"]
        df_5["high"] = ohlc["high"]
        df_5["low"] = ohlc["low"]
        df_5["close"] = ohlc["close"]

        if "volume" in df_1m.columns:
            df_5["volume"] = df_1m["volume"].resample("5min").sum()
        if "value" in df_1m.columns:
            df_5["value"] = df_1m["value"].resample("5min").sum()

        return df_5.dropna(how="all")


# ── Module-level helpers ───────────────────────────────────────────────────

def _iss_get_all(
    session: requests.Session, url: str, query: dict
) -> dict[str, list[dict]]:
    """Use ``ISSClient.get_all()`` for cursor-paginated ISS requests."""
    from apimoex import ISSClient

    client = ISSClient(session, url, query)
    return client.get_all()


def _fetch_json_page(session: requests.Session, url: str) -> Optional[dict]:
    """GET a single ISS JSON page and return the parsed dict."""
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except RequestException as e:
        logger.warning("JSON page fetch failed: %s → %s", url, e)
        return None


# ── Data processing ────────────────────────────────────────────────────────

def _compute_amount(df: pd.DataFrame, lot_size: int) -> pd.DataFrame:
    """Add amount = close * volume * lot_size column."""
    df = df.copy()
    df["amount"] = df["close"].astype("float64") * df["volume"].astype("float64") * lot_size
    return df


def add_amount_column(ticker_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add amount column for all tickers based on their lot size."""
    result = {}
    for ticker, df in ticker_data.items():
        if df.empty:
            result[ticker] = df
            continue
        lot_size = LOT_SIZES.get(ticker, 1) if not _is_macro(ticker) else 1
        result[ticker] = _compute_amount(df, lot_size)
    return result


def save_parquets(
    ticker_data: dict[str, pd.DataFrame],
    output_dir: str = "data/v3/raw",
) -> None:
    """Save per-ticker parquet files and a manifest.json."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = {}

    for ticker, df in ticker_data.items():
        if df.empty:
            meta[ticker] = {"rows": 0, "error": "no data"}
            continue

        cols = ["timestamp", "open", "high", "low", "close", "volume", "amount"]
        df_reset = df.reset_index()

        rename_map = {}
        if "begin" in df_reset.columns:
            rename_map["begin"] = "timestamp"
        elif "index" in df_reset.columns:
            rename_map["index"] = "timestamp"

        if rename_map:
            df_reset = df_reset.rename(columns=rename_map)

        keep = [c for c in cols if c in df_reset.columns]
        save = df_reset[keep].sort_values("timestamp").reset_index(drop=True)

        path = out / f"{ticker}.parquet"
        save.to_parquet(path, index=False)
        meta[ticker] = {
            "rows": len(save),
            "start": str(save["timestamp"].iloc[0]),
            "end": str(save["timestamp"].iloc[-1]),
        }

    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("Manifest written: %s", manifest_path)


# ── Progress tracking (resume support) ─────────────────────────────────────

def _load_progress(progress_path: Path) -> set[str]:
    """Load already-downloaded ('ticker:start') keys."""
    if not progress_path.exists():
        return set()
    with open(progress_path) as f:
        return set(line.strip() for line in f if line.strip())


def _save_progress(progress_path: Path, key: str) -> None:
    """Append a completed key to the progress file."""
    with open(progress_path, "a") as f:
        f.write(key + "\n")


# ── Orchestrator (parallel fetch + process) ────────────────────────────────

def fetch_all(
    fetcher: Optional[Fetcher] = None,
    start_date: str = "2023-01-01",
    end_date: Optional[str] = None,
    interval: int = 10,
    resume: bool = True,
    progress_dir: str = "data/v3/progress",
) -> dict[str, pd.DataFrame]:
    """Fetch all tickers in parallel, chunked by month.

    Args:
        fetcher: A ``Fetcher`` instance. Created with defaults if ``None``.
        start_date: YYYY-MM-DD.
        end_date: YYYY-MM-DD (default: today).
        interval: Candle interval in minutes.
        resume: Skip already-fetched chunk-ticker pairs.
        progress_dir: Directory for progress tracking files.

    Returns:
        ``dict[ticker] → DataFrame`` indexed by ``begin`` (DatetimeIndex, MSK).
    """
    if fetcher is None:
        fetcher = Fetcher()

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    all_tickers: list[str] = list(TICKERS) + list(MACRO_TICKERS)
    chunks: list[tuple[str, str]] = _month_chunks(start_date, end_date)

    print(
        f"Tickers: {len(all_tickers)}, months: {len(chunks)} "
        f"({start_date} → {end_date})"
    )

    progress_path = (
        Path(progress_dir) / "fetched_chunks.txt" if resume else None
    )
    if progress_path:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        done_keys = _load_progress(progress_path)
        print(f"  Already done: {len(done_keys)} chunk-ticker pairs")
    else:
        done_keys: set[str] = set()

    result: dict[str, list[pd.DataFrame]] = {t: [] for t in all_tickers}
    total_reqs = len(chunks) * len(all_tickers)
    completed = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        for chunk_idx, (cs, ce) in enumerate(chunks):
            pending: dict[object, tuple[str, str]] = {}
            for ticker in all_tickers:
                key = f"{ticker}:{cs}"
                if key in done_keys:
                    completed += 1
                    continue
                fut = executor.submit(
                    fetcher.fetch_candles, ticker, interval, cs, ce
                )
                pending[fut] = (ticker, key)

            if not pending:
                continue

            for future in as_completed(pending):
                ticker, key = pending[future]
                try:
                    df = future.result()
                    if df is not None and len(df) > 0:
                        result[ticker].append(df)
                    if progress_path and key not in done_keys:
                        _save_progress(progress_path, key)
                except Exception as e:
                    print(f"  [ERROR] {key}: {e}")

                completed += 1
                if completed % 50 == 0:
                    print(f"  Progress: {completed}/{total_reqs} requests")

    combined: dict[str, pd.DataFrame] = {}
    for ticker in all_tickers:
        if result[ticker]:
            df = pd.concat(result[ticker])
            df = df[~df.index.duplicated(keep="first")].sort_index()
            combined[ticker] = df
            print(f"  {ticker}: {len(df)} bars")
        else:
            combined[ticker] = pd.DataFrame()
            print(f"  {ticker}: NO DATA")

    return combined


def fetch_and_save(
    start_date: str = "2023-01-01",
    end_date: Optional[str] = None,
    output_dir: str = "data/v3/raw",
    resume: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch → add amount → save parquet. Main entry point.

    Example:
        >>> from src.data.fetcher import fetch_and_save
        >>> data = fetch_and_save("2023-01-01", "2026-05-01")
    """
    fetcher = Fetcher()
    raw = fetch_all(fetcher, start_date, end_date, resume=resume)
    with_amount = add_amount_column(raw)
    save_parquets(with_amount, output_dir)
    return with_amount


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MOEX ISS → parquet (Kronos M-FETCH)",
        epilog=(
            "Example: python -m src.data.fetcher "
            "--start 2023-01-01 --end 2026-05-01"
        ),
    )
    parser.add_argument("--start", default="2023-01-01", help="YYYY-MM-DD")
    parser.add_argument(
        "--end", default=None, help="YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--output", default="data/v3/raw", help="output directory"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="continue interrupted fetch (default: on)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="disable resume (start fresh)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print plan only"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="print manifest.json summary and exit",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="candle interval in minutes (default: 10)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.status:
        manifest_path = Path(args.output) / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                data = json.load(f)
            total = sum(v["rows"] for v in data.values())
            tickers_with_data = sum(
                1 for v in data.values() if v["rows"] > 0
            )
            print(
                f"Status: {tickers_with_data}/{len(data)} tickers, "
                f"{total} total bars"
            )
            for t, v in data.items():
                if v["rows"]:
                    status_str = (
                        f"{v['rows']} bars ({v['start']} → {v['end']})"
                    )
                else:
                    status_str = "NO DATA"
                print(f"  {t}: {status_str}")
        else:
            print(f"No manifest at {manifest_path}")
        return

    end = args.end or datetime.now().strftime("%Y-%m-%d")

    chunks = _month_chunks(args.start, end)
    all_tickers = list(TICKERS) + list(MACRO_TICKERS)
    print(
        f"Plan: {len(all_tickers)} tickers × {len(chunks)} months = "
        f"{len(all_tickers) * len(chunks)} requests"
    )
    print(f"  Tickers: {', '.join(all_tickers)}")
    print(f"  Output:  {args.output}/")
    print(f"  Resume:  {'on' if args.resume else 'off'}")
    print(f"  Interval: {args.interval} min")

    if args.dry_run:
        return

    try:
        fetch_and_save(args.start, end, args.output, resume=args.resume)
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run with --resume to continue.")
        exit(1)
    except Exception as e:
        logger.exception("Fatal error during fetch")
        print(f"Fatal error: {e}")
        exit(2)


if __name__ == "__main__":
    main()
