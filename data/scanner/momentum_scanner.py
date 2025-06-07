"""Sniper-focused Momentum Scanner with 3-component scoring system."""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, time
import pandas as pd
import numpy as np
from pathlib import Path
import databento as db
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from data.utils.helpers import ensure_timezone_aware
from config.config import ScannerConfig


class MomentumScanner:
    """Sniper-focused momentum scanner with 3-component scoring and session-aware volume profiling."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        scanner_config: Optional[ScannerConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the momentum scanner.

        Args:
            data_dir: Directory containing Databento files
            output_dir: Directory to save index files
            scanner_config: Consolidated scanner configuration
            logger: Optional logger instance
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logger or logging.getLogger(__name__)

        # Configuration with defaults
        self.config = scanner_config or ScannerConfig()

        # Index paths
        self.day_index_path = self.output_dir / "momentum_days.parquet"
        self.reset_index_path = self.output_dir / "reset_points.parquet"

        # Session-aware volume caches
        self._session_volume_cache: Dict[
            str, Dict[str, float]
        ] = {}  # symbol -> session -> volume

        # Market session definitions
        self._market_sessions = {
            "premarket": (
                self._parse_time(self.config.premarket_start),
                self._parse_time(self.config.premarket_end),
            ),
            "regular": (
                self._parse_time(self.config.regular_start),
                self._parse_time(self.config.regular_end),
            ),
            "postmarket": (
                self._parse_time(self.config.postmarket_start),
                self._parse_time(self.config.postmarket_end),
            ),
        }

    def _parse_time(self, time_str: str) -> time:
        """Parse time string to time object."""
        return datetime.strptime(time_str, "%H:%M").time()

    def scan_all_symbols(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_workers: int = 4,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scan all symbols for momentum days and reset points.

        Args:
            symbols: List of symbols to scan (None for all)
            start_date: Start date for scanning
            end_date: End date for scanning
            max_workers: Number of parallel workers

        Returns:
            Tuple of (day_index_df, reset_points_df)
        """
        # Discover available symbols if not provided
        if symbols is None:
            symbols = self._discover_symbols()

        self.logger.info(f"Scanning {len(symbols)} symbols for momentum days...")

        # Process symbols in parallel
        all_day_records = []
        all_reset_records = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit scanning tasks
            future_to_symbol = {
                executor.submit(self._scan_symbol, symbol, start_date, end_date): symbol
                for symbol in symbols
            }

            # Process results as they complete
            with tqdm(total=len(symbols), desc="Scanning symbols") as pbar:
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        day_records, reset_records = future.result()
                        all_day_records.extend(day_records)
                        all_reset_records.extend(reset_records)
                    except Exception as e:
                        self.logger.error(f"Error scanning {symbol}: {e}")
                    finally:
                        pbar.update(1)

        # Create DataFrames
        day_df = pd.DataFrame(all_day_records)
        reset_df = pd.DataFrame(all_reset_records)

        # Save indices
        if not day_df.empty:
            day_df.to_parquet(self.day_index_path)
            self.logger.info(
                f"Saved {len(day_df)} momentum days to {self.day_index_path}"
            )

        if not reset_df.empty:
            reset_df.to_parquet(self.reset_index_path)
            self.logger.info(
                f"Saved {len(reset_df)} reset points to {self.reset_index_path}"
            )

        return day_df, reset_df

    def _discover_symbols(self) -> List[str]:
        """Discover all available symbols from data directory."""
        symbols = set()

        # Walk through data directory looking for metadata files
        for metadata_path in self.data_dir.rglob("metadata.json"):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    query_symbols = metadata.get("query", {}).get("symbols", [])
                    symbols.update(str(s).upper() for s in query_symbols)
            except Exception as e:
                self.logger.warning(f"Error reading {metadata_path}: {e}")

        return sorted(list(symbols))

    def filter_reset_points_by_ranges(
        self,
        reset_points: List[Dict],
        roc_range: List[float],
        activity_range: List[float],
    ) -> List[Dict]:
        """Filter reset points using direct range criteria"""
        filtered_points = []
        for reset_point in reset_points:
            roc_score = reset_point.get("roc_score", 0.0)
            activity_score = reset_point.get("activity_score", 0.0)

            # Check if scores fall within ranges
            if (
                roc_range[0] <= roc_score <= roc_range[1]
                and activity_range[0] <= activity_score <= activity_range[1]
            ):
                filtered_points.append(reset_point)

        return filtered_points

    def _scan_symbol(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Scan a single symbol for momentum days and reset points.

        Returns:
            Tuple of (day_records, reset_records)
        """
        day_records = []
        reset_records = []

        # Get 1-second OHLCV data for the symbol
        ohlcv_files = self._find_ohlcv_files(symbol, "1s")
        if not ohlcv_files:
            self.logger.warning(f"No 1s OHLCV data found for {symbol}")
            return day_records, reset_records

        # Get session-aware volumes for comparison
        session_volumes = self._get_session_volumes(symbol)

        # Process each file (typically contains multiple days)
        for file_path in ohlcv_files:
            try:
                # Load data
                store = db.DBNStore.from_file(file_path)
                df = store.to_df()

                if df.empty:
                    continue

                # Ensure proper datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")

                # Filter by date range if provided
                if start_date:
                    df = df[df.index >= ensure_timezone_aware(start_date)]
                if end_date:
                    df = df[df.index <= ensure_timezone_aware(end_date)]

                if df.empty:
                    continue

                # Group by date and analyze each day
                for date, day_data in df.groupby(df.index.date):
                    day_record, day_reset_points = self._analyze_day(
                        symbol, date, day_data, session_volumes
                    )

                    if day_record:
                        day_records.append(day_record)
                        reset_records.extend(day_reset_points)

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")

        return day_records, reset_records

    def _analyze_day(
        self,
        symbol: str,
        date: pd.Timestamp,
        day_data: pd.DataFrame,
        session_volumes: Dict[str, float],
    ) -> Tuple[Optional[Dict], List[Dict]]:
        """Analyze a single day for momentum characteristics using sniper approach.

        Returns:
            Tuple of (day_record, reset_points) or (None, [])
        """
        # Filter to trading hours (4 AM - 8 PM ET)
        day_data = self._filter_trading_hours(day_data)
        if len(day_data) < 100:  # Need sufficient data points
            return None, []

        # Calculate daily metrics
        open_price = day_data["open"].iloc[0]
        close_price = day_data["close"].iloc[-1]
        high_price = day_data["high"].max()
        low_price = day_data["low"].min()
        total_volume = day_data["volume"].sum()

        # Calculate price movement
        daily_return = (close_price - open_price) / open_price
        max_move_up = (high_price - open_price) / open_price
        max_move_down = (open_price - low_price) / open_price
        max_intraday_move = max(max_move_up, max_move_down)

        # Remove direction-based filtering - capture all volatility

        # Calculate average session volume for comparison
        avg_session_volume = np.mean(list(session_volumes.values()))

        # Check momentum criteria (NO CAPS - capture all volatility)
        is_momentum_day = (
            max_intraday_move >= self.config.min_daily_move
            and total_volume >= avg_session_volume * self.config.min_volume_multiplier
        )

        if not is_momentum_day:
            return None, []

        # Find halts
        halt_count = self._count_halts(symbol, date)

        # Calculate simple activity score for day ranking (no caps)
        volume_multiplier = total_volume / avg_session_volume
        activity_score = (
            np.tanh(max_intraday_move * 10) * 0.5
            + np.tanh(volume_multiplier / 10) * 0.5
        )

        # Create day record
        day_record = {
            "symbol": symbol,
            "date": pd.Timestamp(date),
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": total_volume,
            "daily_return": daily_return,
            "max_intraday_move": max_intraday_move,
            "volume_multiplier": volume_multiplier,
            "halt_count": halt_count,
            "activity_score": activity_score,
            "file_paths": [str(f) for f in self._find_day_files(symbol, date)],
        }

        # Find reset points using 2-component scoring (ROC + Activity)
        reset_points = self._find_reset_points_2component(
            symbol, date, day_data, session_volumes
        )

        return day_record, reset_points

    def _find_reset_points_2component(
        self,
        symbol: str,
        date: pd.Timestamp,
        day_data: pd.DataFrame,
        session_volumes: Dict[str, float],
    ) -> List[Dict]:
        """Find reset points using 2-component scoring (ROC + Activity) every 5 minutes."""
        reset_points = []

        # Resample to 5-minute bars for reset point generation
        five_min_bars = (
            day_data.resample("5min")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        if len(five_min_bars) < 12:  # Need at least 1 hour of data
            return reset_points

        # Add session information
        five_min_bars["session"] = five_min_bars.apply(
            lambda row: self._get_market_session(pd.Timestamp(row.name)), axis=1
        )

        # Calculate 2-component scores within session context
        five_min_bars = self._calculate_2component_scores(
            five_min_bars, session_volumes
        )

        if len(five_min_bars) < self.config.min_reset_points:
            self.logger.warning(
                f"Only {len(five_min_bars)} valid reset points for {symbol} {date}"
            )
            return reset_points

        # Calculate combined score for weighting (scale ROC to be more significant)
        # Since ROC is typically small (0-0.3 range), scale it up to balance with activity
        roc_scaled = five_min_bars["roc_score"].abs() * 10.0
        five_min_bars["combined_score"] = (
            roc_scaled * self.config.roc_weight
            + five_min_bars["activity_score"] * self.config.activity_weight
        )

        # Generate reset points - each 5-minute bar is a potential reset point
        reset_points = self._generate_2component_reset_points(
            five_min_bars, symbol, date
        )

        return reset_points

    def _calculate_2component_scores(
        self, five_min_bars: pd.DataFrame, session_volumes: Dict[str, float]
    ) -> pd.DataFrame:
        """Calculate ROC [-1.0, 1.0] and activity [0.0, 1.0] scores using rolling window method."""
        return self._calculate_rolling_scores(five_min_bars, session_volumes)

    def _calculate_rolling_scores(
        self, five_min_bars: pd.DataFrame, session_volumes: Dict[str, float]
    ) -> pd.DataFrame:
        """Calculate rolling window scores: ROC [-1.0, 1.0], Activity [0.0, 1.0]."""

        # Configuration
        window_size_periods = int(60 / 5)  # 60 minutes / 5-min periods (not used)
        roc_periods = int(
            self.config.roc_lookback_minutes / 5
        )  # Convert to 5-min periods
        activity_periods = int(
            self.config.activity_lookback_minutes / 5
        )  # Convert to 5-min periods

        # Group by session for session-aware calculations
        session_scores = []

        for session in ["premarket", "regular", "postmarket"]:
            session_data = five_min_bars[five_min_bars["session"] == session].copy()
            if len(session_data) == 0:
                continue

            # 1. ROC Score [-1.0, 1.0]: Directional rate of change
            session_data["price_change_roc"] = session_data["close"].pct_change(
                roc_periods
            )

            # Clip to [-1.0, 1.0] range to represent actual percentage change
            session_data["roc_score"] = np.clip(
                session_data["price_change_roc"], -1.0, 1.0
            )

            # 2. Activity Score [0.0, 1.0]: Improved volume normalization
            session_data["volume_rolling"] = (
                session_data["volume"].rolling(activity_periods, min_periods=1).mean()
            )

            # Get session baseline volume
            session_baseline = session_volumes.get(
                session, session_volumes.get("regular", 1000)
            )

            # Better normalization using tanh for smooth [0.0, 1.0] mapping
            volume_ratio = session_data["volume_rolling"] / session_baseline

            # Apply volume significance threshold (use reasonable default)
            min_volume_threshold = 1000.0  # Minimum volume threshold
            volume_significant = session_data["volume_rolling"] >= min_volume_threshold

            # Use tanh normalization for better distribution in [0.0, 1.0]
            # tanh(x) maps to [-1, 1], so (tanh(x) + 1) / 2 maps to [0, 1]
            normalized_volume = (np.tanh(volume_ratio - 1) + 1) / 2

            # Apply neutral score for low volume periods
            session_data["activity_score"] = np.where(
                volume_significant,
                normalized_volume,
                0.5,  # Neutral score
            )

            # Store volume deviation for compatibility
            session_data["volume_deviation"] = volume_ratio

            session_scores.append(session_data)

        # Combine all sessions
        if session_scores:
            result = pd.concat(session_scores).sort_index()
        else:
            result = five_min_bars.copy()
            result["roc_score"] = 0.0  # Neutral directional score
            result["activity_score"] = 0.5  # Neutral activity score
            result["price_change_roc"] = 0.0
            result["volume_deviation"] = 1.0

        # Fill NaN values with neutral scores
        result["roc_score"] = result["roc_score"].fillna(0.0)  # Neutral for directional
        result["activity_score"] = result["activity_score"].fillna(
            0.5
        )  # Neutral for activity

        return result

    def _generate_2component_reset_points(
        self, five_min_bars: pd.DataFrame, symbol: str, date: pd.Timestamp
    ) -> List[Dict]:
        """Generate reset points using 2-component scoring every 5 minutes."""
        reset_points = []

        # Clean data
        five_min_bars = five_min_bars.dropna(subset=["combined_score"])
        if len(five_min_bars) == 0:
            return reset_points

        # All 5-minute bars become reset points - no sampling needed
        for idx, row in five_min_bars.iterrows():
            reset_points.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "timestamp": row.name,
                    "roc_score": row["roc_score"],
                    "activity_score": row["activity_score"],
                    "combined_score": row["combined_score"],
                    "price": row["close"],
                    "volume": row["volume"],
                    "session": row["session"],
                    "price_change_roc": row.get("price_change_roc", 0.0),
                    "volume_deviation": row.get("volume_deviation", 0.0),
                }
            )

        # Sort by timestamp
        reset_points.sort(key=lambda x: x["timestamp"])

        return reset_points

    def _is_within_trading_hours(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is within trading hours (4 AM - 8 PM ET)."""
        # Convert to ET for checking
        try:
            et_time = timestamp.tz_convert("US/Eastern").time()
            return (
                pd.Timestamp("04:00").time() <= et_time <= pd.Timestamp("20:00").time()
            )
        except:
            # Fallback for timezone issues
            return True

    def _filter_trading_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data to regular trading hours (4 AM - 8 PM ET)."""
        # Convert to ET timezone for filtering
        df_et = df.copy()
        df_et.index = df_et.index.tz_convert("America/New_York")

        # Filter to 4 AM - 8 PM ET (inclusive of 8 PM hour)
        mask = (df_et.index.hour >= 4) & (df_et.index.hour <= 20)
        return df[mask]

    def _get_session_volumes(self, symbol: str) -> Dict[str, float]:
        """Get session-aware average volumes for a symbol."""
        if symbol in self._session_volume_cache:
            return self._session_volume_cache[symbol]

        session_volumes = {
            "premarket": 500_000,
            "regular": 2_000_000,
            "postmarket": 300_000,
        }  # Defaults

        # Calculate from 1-minute or 1-second data if available
        for timeframe in ["1m", "1s"]:
            files = self._find_ohlcv_files(symbol, timeframe)
            if not files:
                continue

            try:
                all_data = []
                for file_path in files[-self.config.volume_window_days :]:
                    store = db.DBNStore.from_file(file_path)
                    df = store.to_df()
                    if not df.empty and "volume" in df.columns:
                        # Ensure proper datetime index
                        if not isinstance(df.index, pd.DatetimeIndex):
                            df.index = pd.to_datetime(df.index)
                        if df.index.tz is None:
                            df.index = df.index.tz_localize("UTC")
                        all_data.append(df)

                if all_data:
                    combined_df = pd.concat(all_data)
                    session_volumes = self._calculate_session_volumes(combined_df)
                    break

            except Exception as e:
                self.logger.warning(
                    f"Error calculating session volumes for {symbol}: {e}"
                )

        self._session_volume_cache[symbol] = session_volumes
        return session_volumes

    def _calculate_session_volumes(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate median volumes by market session."""
        session_volumes = {}

        # Add session column
        df_et = df.copy()
        df_et.index = df_et.index.tz_convert("America/New_York")
        df_et["session"] = df_et.apply(
            lambda row: self._get_market_session(pd.Timestamp(row.name)), axis=1
        )

        # Calculate session medians
        for session in ["premarket", "regular", "postmarket"]:
            session_data = df_et[df_et["session"] == session]
            if not session_data.empty:
                session_volumes[session] = session_data["volume"].median()
            else:
                # Fallback values
                session_volumes[session] = {
                    "premarket": 500_000,
                    "regular": 2_000_000,
                    "postmarket": 300_000,
                }[session]

        return session_volumes

    def _get_market_session(self, timestamp: pd.Timestamp) -> str:
        """Determine market session for a timestamp."""
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize("UTC")
        et_time = timestamp.tz_convert("America/New_York").time()

        for session, (start_time, end_time) in self._market_sessions.items():
            if start_time <= et_time < end_time:
                return session

        return "closed"

    def _count_halts(self, symbol: str, date: pd.Timestamp) -> int:
        """Count trading halts for a symbol on a given date."""
        halt_count = 0

        # Look for status files
        status_files = self._find_status_files(symbol, date)
        for file_path in status_files:
            try:
                store = db.DBNStore.from_file(file_path)
                df = store.to_df()

                # Filter to the specific date
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")

                df = df[df.index.date == date]

                # Count halt events
                if "action" in df.columns:
                    # Databento status codes: 8=HALTED, 9=PAUSED, 10=SUSPENDED
                    halt_count += df["action"].isin([8, 9, 10]).sum()

            except Exception as e:
                self.logger.warning(f"Error counting halts from {file_path}: {e}")

        return halt_count

    def _find_ohlcv_files(self, symbol: str, timeframe: str) -> List[Path]:
        """Find OHLCV files for a symbol and timeframe."""
        files = []

        # First try with symbol in filename
        pattern = f"*{symbol.lower()}*.ohlcv-{timeframe}.dbn*"
        for file_path in self.data_dir.rglob(pattern):
            files.append(file_path)

        # If no files found, try generic pattern and verify symbol
        if not files:
            pattern = f"*.ohlcv-{timeframe}.dbn*"
            for file_path in self.data_dir.rglob(pattern):
                if self._file_contains_symbol(file_path, symbol):
                    files.append(file_path)

        return sorted(files)

    def _find_status_files(self, symbol: str, date: pd.Timestamp) -> List[Path]:
        """Find status files for a symbol on a date."""
        files = []
        date_str = date.strftime("%Y%m%d")
        pattern = f"*{date_str}*.status.dbn*"

        for file_path in self.data_dir.rglob(pattern):
            # Verify it's for the right symbol
            if self._file_contains_symbol(file_path, symbol):
                files.append(file_path)

        return files

    def _find_day_files(self, symbol: str, date: pd.Timestamp) -> List[Path]:
        """Find all data files for a symbol on a specific date."""
        files = []
        date_str = date.strftime("%Y%m%d")

        # Look for files with the date in the name
        for pattern in [f"*{date_str}*.dbn*", f"*{symbol.lower()}*.dbn*"]:
            for file_path in self.data_dir.rglob(pattern):
                if self._file_contains_symbol(file_path, symbol):
                    files.append(file_path)

        return list(set(files))  # Remove duplicates

    def _file_contains_symbol(self, file_path: Path, symbol: str) -> bool:
        """Check if a file contains data for a specific symbol."""
        # Check parent directory for metadata
        metadata_path = file_path.parent / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    symbols = metadata.get("query", {}).get("symbols", [])
                    return symbol.upper() in [str(s).upper() for s in symbols]
            except:
                pass

        # Fallback to filename check
        return symbol.lower() in str(file_path).lower()

    def load_index(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load existing momentum indices from disk.

        Returns:
            Tuple of (day_index_df, reset_points_df)
        """
        day_df = pd.DataFrame()
        reset_df = pd.DataFrame()

        if self.day_index_path.exists():
            day_df = pd.read_parquet(self.day_index_path)
            self.logger.info(f"Loaded {len(day_df)} momentum days from index")

        if self.reset_index_path.exists():
            reset_df = pd.read_parquet(self.reset_index_path)
            self.logger.info(f"Loaded {len(reset_df)} reset points from index")

        return day_df, reset_df

    def query_momentum_days(
        self,
        symbol: str,
        min_activity: float = 0.5,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Query momentum days for a symbol.

        Args:
            symbol: Symbol to query
            min_activity: Minimum activity score
            start_date: Start date filter
            end_date: End date filter

        Returns:
            DataFrame of matching momentum days
        """
        day_df, _ = self.load_index()

        if day_df.empty:
            return pd.DataFrame()

        # Filter by symbol
        mask = day_df["symbol"] == symbol.upper()

        # Filter by activity score
        mask &= day_df["activity_score"] >= min_activity

        # Remove direction filtering

        # Filter by date range
        if start_date:
            mask &= day_df["date"] >= pd.Timestamp(start_date)
        if end_date:
            mask &= day_df["date"] <= pd.Timestamp(end_date)

        return day_df[mask].sort_values("activity_score", ascending=False)

    def query_reset_points(
        self,
        symbol: str,
        date: Optional[pd.Timestamp] = None,
        min_activity: float = 0.5,
    ) -> pd.DataFrame:
        """Query reset points for a symbol.

        Args:
            symbol: Symbol to query
            date: Specific date (None for all)
            min_activity: Minimum combined activity score

        Returns:
            DataFrame of matching reset points
        """
        _, reset_df = self.load_index()

        if reset_df.empty:
            return pd.DataFrame()

        # Filter by symbol
        mask = reset_df["symbol"] == symbol.upper()

        # Filter by date
        if date:
            mask &= reset_df["date"] == date

        # Filter by activity
        mask &= reset_df["combined_score"] >= min_activity

        # Remove direction filtering

        return reset_df[mask].sort_values("combined_score", ascending=False)

    def get_top_momentum_days(
        self,
        symbol: str,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Get top N momentum days by activity score.

        Args:
            symbol: Symbol to query
            top_n: Number of top days to return

        Returns:
            DataFrame of top momentum days
        """
        days = self.query_momentum_days(symbol, min_activity=0.0)
        return days.head(top_n)

    def get_momentum_day_summary(self) -> pd.DataFrame:
        """Get summary statistics for all momentum days.

        Returns:
            DataFrame with summary by symbol
        """
        day_df, _ = self.load_index()

        if day_df.empty:
            return pd.DataFrame()

        summary = day_df.groupby("symbol").agg(
            {
                "date": "count",
                "activity_score": ["mean", "max"],
                "max_intraday_move": ["mean", "max"],
                "volume_multiplier": ["mean", "max"],
            }
        )

        summary.columns = ["_".join(col).strip() for col in summary.columns.values]
        summary = summary.rename(
            columns={
                "date_count": "total_days",
                "activity_score_mean": "avg_activity",
                "activity_score_max": "max_activity",
                "max_intraday_move_mean": "avg_move",
                "max_intraday_move_max": "max_move",
                "volume_multiplier_mean": "avg_volume_mult",
                "volume_multiplier_max": "max_volume_mult",
            }
        )

        return summary.sort_values("total_days", ascending=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scan momentum days in Databento files"
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.5,
        help="Minimum quality score (default: 0.5)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=[],
        help="Symbols to scan (default: scan all available)",
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="Force rebuild existing indices"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize scanner with default paths
    data_dir = "dnb"
    output_dir = "cache/indices/momentum_index"
    scanner = MomentumScanner(data_dir=data_dir, output_dir=output_dir)

    try:
        # Scan symbols
        logger.info(f"Scanning momentum days for symbols: {args.symbols}")
        day_df, reset_df = scanner.scan_all_symbols(symbols=args.symbols)

        # Show results
        logger.info(f"Loading momentum days with min quality {args.min_quality}...")

        for symbol in args.symbols:
            days = scanner.query_momentum_days(symbol, min_activity=args.min_quality)
            reset_points = scanner.query_reset_points(
                symbol, min_activity=args.min_quality
            )

            logger.info(f"\n{symbol} Results:")
            logger.info(f"  Momentum days: {len(days)}")
            logger.info(f"  Reset points: {len(reset_points)}")

            if len(days) > 0:
                logger.info(
                    f"  Quality range: {days['activity_score'].min():.3f} - {days['activity_score'].max():.3f}"
                )

    except Exception as e:
        logger.error(f"Error during scanning: {e}")
        raise
