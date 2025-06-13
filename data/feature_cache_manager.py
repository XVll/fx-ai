"""Feature Cache Manager - Persistent caching for on-demand features

This manager provides intelligent caching of extracted features to disk, with:
- Timestamp-aware caching for partial day coverage
- Automatic cache invalidation when feature structure changes
- Dynamic loading/unloading when switching between days
- Unified cache organization under cache/ folder

Designed for momentum training where the same day is trained 10+ times with
different reset points, providing massive speed improvements for repeated sessions.
"""

import pickle
import hashlib
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Set, Tuple
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass
import json
from feature.simple_feature_manager import SimpleFeatureManager
from core.path_manager import get_path_manager
from core.shutdown import IShutdownHandler, get_global_shutdown_manager


@dataclass
class CacheMetadata:
    """Metadata for a feature cache file"""

    symbol: str
    date: date
    feature_hash: str
    cached_timestamps: Set[pd.Timestamp]
    feature_dimensions: Dict[str, Tuple[int, int]]  # category -> (seq_len, feat_dim)
    created_at: datetime
    last_accessed: datetime
    version: str = "1.0"


class FeatureCacheManager(IShutdownHandler):
    """Manages persistent feature caching with automatic invalidation and loading"""

    def __init__(
        self, cache_dir: Optional[str] = None, logger: Optional[logging.Logger] = None
    ):
        # Use PathManager for cache directory
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        else:
            path_manager = get_path_manager()
            self.cache_dir = path_manager.features_cache_dir
        
        self.logger = logger or logging.getLogger(__name__)

        # Ensure cache directory exists (PathManager handles this, but double-check)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Current session state
        self._current_cache: Optional[Dict[str, Any]] = None
        self._current_metadata: Optional[CacheMetadata] = None
        self._current_session: Optional[Tuple[str, date]] = None
        self._dirty_timestamps: Set[pd.Timestamp] = set()
        
        # Shutdown state
        self._shutdown_requested: bool = False

        self.logger.info(
            f"FeatureCacheManager initialized with cache_dir: {self.cache_dir}"
        )

    def _compute_feature_hash(self, feature_manager: SimpleFeatureManager) -> str:
        """Compute hash of feature structure to detect changes"""
        try:
            # Get all enabled features and their structure
            feature_info = {}

            for category in ["hf", "mf", "lf"]:
                enabled_features = feature_manager.get_enabled_features(category)
                feature_info[category] = {
                    "features": sorted(enabled_features),
                    "count": len(enabled_features),
                }

            # Add dimensions
            feature_info["dimensions"] = {
                "hf": (feature_manager.hf_seq_len, feature_manager.hf_feat_dim),
                "mf": (feature_manager.mf_seq_len, feature_manager.mf_feat_dim),
                "lf": (feature_manager.lf_seq_len, feature_manager.lf_feat_dim),
            }

            # Create hash from the structure
            feature_str = json.dumps(feature_info, sort_keys=True)
            return hashlib.md5(feature_str.encode()).hexdigest()

        except Exception as e:
            self.logger.warning(f"Failed to compute feature hash: {e}")
            return "unknown"

    def _get_cache_file_path(self, symbol: str, cache_date: date) -> Path:
        """Get cache file path for symbol and date"""
        return self.cache_dir / symbol / f"{cache_date.strftime('%Y-%m-%d')}.pkl"

    def _get_metadata_file_path(self, symbol: str, cache_date: date) -> Path:
        """Get metadata file path for symbol and date"""
        return self.cache_dir / symbol / f"{cache_date.strftime('%Y-%m-%d')}_meta.json"

    def _load_metadata(self, symbol: str, cache_date: date) -> Optional[CacheMetadata]:
        """Load cache metadata from file"""
        try:
            metadata_path = self._get_metadata_file_path(symbol, cache_date)

            if not metadata_path.exists():
                return None

            with open(metadata_path, "r") as f:
                data = json.load(f)

            # Convert timestamps back from strings
            cached_timestamps = {pd.Timestamp(ts) for ts in data["cached_timestamps"]}

            return CacheMetadata(
                symbol=data["symbol"],
                date=datetime.fromisoformat(data["date"]).date(),
                feature_hash=data["feature_hash"],
                cached_timestamps=cached_timestamps,
                feature_dimensions=data["feature_dimensions"],
                created_at=datetime.fromisoformat(data["created_at"]),
                last_accessed=datetime.fromisoformat(data["last_accessed"]),
                version=data.get("version", "1.0"),
            )

        except Exception as e:
            self.logger.warning(
                f"Failed to load metadata for {symbol} {cache_date}: {e}"
            )
            return None

    def _save_metadata(self, metadata: CacheMetadata) -> bool:
        """Save cache metadata to file"""
        try:
            metadata_path = self._get_metadata_file_path(metadata.symbol, metadata.date)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert timestamps to strings for JSON serialization
            data = {
                "symbol": metadata.symbol,
                "date": metadata.date.isoformat(),
                "feature_hash": metadata.feature_hash,
                "cached_timestamps": [
                    ts.isoformat() for ts in metadata.cached_timestamps
                ],
                "feature_dimensions": metadata.feature_dimensions,
                "created_at": metadata.created_at.isoformat(),
                "last_accessed": metadata.last_accessed.isoformat(),
                "version": metadata.version,
            }

            with open(metadata_path, "w") as f:
                json.dump(data, f, indent=2)

            return True

        except Exception as e:
            self.logger.error(
                f"Failed to save metadata for {metadata.symbol} {metadata.date}: {e}"
            )
            return False

    def _load_cache_data(
        self, symbol: str, cache_date: date
    ) -> Optional[Dict[str, Any]]:
        """Load cached feature data from file"""
        try:
            cache_path = self._get_cache_file_path(symbol, cache_date)

            if not cache_path.exists():
                return None

            with open(cache_path, "rb") as f:
                return pickle.load(f)

        except Exception as e:
            self.logger.warning(
                f"Failed to load cache data for {symbol} {cache_date}: {e}"
            )
            return None

    def _save_cache_data(
        self, data: Dict[str, Any], symbol: str, cache_date: date
    ) -> bool:
        """Save feature data to cache file"""
        try:
            cache_path = self._get_cache_file_path(symbol, cache_date)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            with open(cache_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            return True

        except Exception as e:
            self.logger.error(
                f"Failed to save cache data for {symbol} {cache_date}: {e}"
            )
            return False

    def _invalidate_cache(self, symbol: str, cache_date: date) -> None:
        """Remove cache files due to invalidation"""
        try:
            cache_path = self._get_cache_file_path(symbol, cache_date)
            metadata_path = self._get_metadata_file_path(symbol, cache_date)

            if cache_path.exists():
                cache_path.unlink()
                self.logger.info(f"Removed cache file: {cache_path}")

            if metadata_path.exists():
                metadata_path.unlink()
                self.logger.info(f"Removed metadata file: {metadata_path}")

        except Exception as e:
            self.logger.error(
                f"Failed to invalidate cache for {symbol} {cache_date}: {e}"
            )

    def load_session(
        self, symbol: str, cache_date: date, feature_manager: SimpleFeatureManager
    ) -> bool:
        """Load cache session for a symbol/date, with automatic invalidation check"""
        try:
            # Check if already loaded
            if self._current_session == (symbol, cache_date):
                self.logger.debug(
                    f"Cache session already loaded for {symbol} {cache_date}"
                )
                return True

            # Unload current session first
            if self._current_session is not None:
                self.unload_session()

            # Compute current feature hash
            current_hash = self._compute_feature_hash(feature_manager)

            # Load metadata
            metadata = self._load_metadata(symbol, cache_date)

            # Check if cache needs invalidation
            if metadata is not None and metadata.feature_hash != current_hash:
                self.logger.info(
                    f"Feature structure changed for {symbol} {cache_date}, invalidating cache"
                )
                self._invalidate_cache(symbol, cache_date)
                metadata = None

            # Load or create cache
            if metadata is None:
                # Create new cache
                self.logger.info(
                    f"Creating new feature cache for {symbol} {cache_date}"
                )

                dimensions = {
                    "hf": (feature_manager.hf_seq_len, feature_manager.hf_feat_dim),
                    "mf": (feature_manager.mf_seq_len, feature_manager.mf_feat_dim),
                    "lf": (feature_manager.lf_seq_len, feature_manager.lf_feat_dim),
                }

                metadata = CacheMetadata(
                    symbol=symbol,
                    date=cache_date,
                    feature_hash=current_hash,
                    cached_timestamps=set(),
                    feature_dimensions=dimensions,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                )

                cache_data = {}  # Empty cache
            else:
                # Load existing cache
                self.logger.info(
                    f"Loading existing feature cache for {symbol} {cache_date} ({len(metadata.cached_timestamps)} timestamps)"
                )
                cache_data = self._load_cache_data(symbol, cache_date) or {}

                # Update last accessed
                metadata.last_accessed = datetime.now()

            # Set current session
            self._current_cache = cache_data
            self._current_metadata = metadata
            self._current_session = (symbol, cache_date)
            self._dirty_timestamps = set()

            self.logger.info(f"Feature cache session loaded for {symbol} {cache_date}")
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to load cache session for {symbol} {cache_date}: {e}"
            )
            return False

    def unload_session(self) -> bool:
        """Unload current cache session, saving any changes"""
        try:
            if self._current_session is None:
                return True

            symbol, cache_date = self._current_session

            # Save if there are changes
            if self._dirty_timestamps and self._current_metadata is not None:
                self.logger.info(
                    f"Saving {len(self._dirty_timestamps)} new cached timestamps for {symbol} {cache_date}"
                )

                # Update metadata
                self._current_metadata.cached_timestamps.update(self._dirty_timestamps)
                self._current_metadata.last_accessed = datetime.now()

                # Save both data and metadata
                success = self._save_cache_data(
                    self._current_cache, symbol, cache_date
                ) and self._save_metadata(self._current_metadata)

                if success:
                    self.logger.info(f"Cache session saved for {symbol} {cache_date}")
                else:
                    self.logger.error(
                        f"Failed to save cache session for {symbol} {cache_date}"
                    )

            # Clear current session
            self._current_cache = None
            self._current_metadata = None
            self._current_session = None
            self._dirty_timestamps = set()

            return True

        except Exception as e:
            self.logger.error(f"Failed to unload cache session: {e}")
            return False

    def get_cached_features(
        self, timestamp: pd.Timestamp
    ) -> Optional[Dict[str, np.ndarray]]:
        """Get cached features for a timestamp"""
        if self._current_cache is None:
            return None

        # Try multiple timestamp formats to handle timezone mismatches
        timestamp_keys = [
            timestamp.isoformat(),  # Original format
        ]

        # Add timezone-aware variants if timestamp is timezone-naive
        if timestamp.tz is None:
            timestamp_keys.append(timestamp.tz_localize("UTC").isoformat())

        # Try each key format
        for timestamp_key in timestamp_keys:
            result = self._current_cache.get(timestamp_key)
            if result is not None:
                self.logger.debug(f"Cache hit for {timestamp_key}")
                return result

        # Debug logging for cache misses
        if self._current_cache:
            self.logger.debug(
                f"Cache miss for {timestamp_keys[0]}. Available keys sample: {list(self._current_cache.keys())[:3]}"
            )

        return None

    def cache_features(
        self, timestamp: pd.Timestamp, features: Dict[str, np.ndarray]
    ) -> None:
        """Cache features for a timestamp"""
        if self._current_cache is None:
            return

        # Ensure consistent timezone handling for storage
        if timestamp.tz is None:
            timestamp_key = timestamp.tz_localize("UTC").isoformat()
            storage_timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp_key = timestamp.isoformat()
            storage_timestamp = timestamp

        self._current_cache[timestamp_key] = features
        self._dirty_timestamps.add(storage_timestamp)

        # Auto-save every 10 cached features to avoid data loss (unless shutdown requested)
        if len(self._dirty_timestamps) >= 10 and not self._shutdown_requested:
            self._save_current_progress()

    def is_timestamp_cached(self, timestamp: pd.Timestamp) -> bool:
        """Check if features are cached for a timestamp"""
        if self._current_metadata is None:
            return False

        return timestamp in self._current_metadata.cached_timestamps

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about current cache session"""
        if self._current_session is None or self._current_metadata is None:
            return {}

        symbol, cache_date = self._current_session

        return {
            "symbol": symbol,
            "date": cache_date.isoformat(),
            "cached_timestamps": len(self._current_metadata.cached_timestamps),
            "dirty_timestamps": len(self._dirty_timestamps),
            "feature_hash": self._current_metadata.feature_hash,
            "last_accessed": self._current_metadata.last_accessed.isoformat(),
            "cache_size_mb": len(str(self._current_cache)) / (1024 * 1024)
            if self._current_cache
            else 0,
        }

    def cleanup_old_caches(self, max_age_days: int = 30) -> int:
        """Remove cache files older than max_age_days"""
        try:
            removed_count = 0
            cutoff_date = datetime.now() - pd.Timedelta(days=max_age_days)

            for symbol_dir in self.cache_dir.iterdir():
                if not symbol_dir.is_dir():
                    continue

                for file_path in symbol_dir.iterdir():
                    if file_path.suffix in [".pkl", ".json"]:
                        try:
                            # Check file modification time
                            if (
                                datetime.fromtimestamp(file_path.stat().st_mtime)
                                < cutoff_date
                            ):
                                file_path.unlink()
                                removed_count += 1
                                self.logger.debug(
                                    f"Removed old cache file: {file_path}"
                                )
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to check/remove file {file_path}: {e}"
                            )

            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old cache files")

            return removed_count

        except Exception as e:
            self.logger.error(f"Failed to cleanup old caches: {e}")
            return 0

    def _save_current_progress(self) -> bool:
        """Save current progress to avoid data loss"""
        if self._current_session is None or not self._dirty_timestamps:
            return True

        symbol, cache_date = self._current_session

        try:
            # Update metadata with current dirty timestamps
            if self._current_metadata is not None:
                self._current_metadata.cached_timestamps.update(self._dirty_timestamps)
                self._current_metadata.last_accessed = datetime.now()

                # Save both data and metadata
                success = self._save_cache_data(
                    self._current_cache, symbol, cache_date
                ) and self._save_metadata(self._current_metadata)

                if success:
                    self.logger.info(
                        f"Auto-saved {len(self._dirty_timestamps)} cached features for {symbol} {cache_date}"
                    )
                    # Clear dirty timestamps since they're now saved
                    self._dirty_timestamps = set()
                    return True
                else:
                    self.logger.error(
                        f"Failed to auto-save cache for {symbol} {cache_date}"
                    )
                    return False

            return False

        except Exception as e:
            self.logger.error(
                f"Exception during auto-save for {symbol} {cache_date}: {e}"
            )
            return False

    def register_shutdown(self) -> None:
        """Register this component with the global shutdown manager."""
        try:
            shutdown_manager = get_global_shutdown_manager()
            shutdown_manager.register_component(
                name="FeatureCacheManager",
                shutdown_func=self.shutdown,
                timeout=10.0
            )
            self.logger.debug("FeatureCacheManager registered with shutdown manager")
        except Exception as e:
            self.logger.warning(f"Failed to register FeatureCacheManager with shutdown manager: {e}")

    def shutdown(self) -> None:
        """Perform graceful shutdown - save any pending cache and stop operations."""
        self.logger.info("🛑 FeatureCacheManager shutdown initiated")
        
        # Set shutdown flag to stop auto-save operations
        self._shutdown_requested = True
        
        try:
            # Save any pending cached features before shutdown
            if self._dirty_timestamps and self._current_session:
                symbol, cache_date = self._current_session
                self.logger.info(f"Saving {len(self._dirty_timestamps)} pending cached features for {symbol} {cache_date}")
                self._save_current_progress()
            
            # Unload current session
            self.unload_session()
            
            self.logger.info("✅ FeatureCacheManager shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during FeatureCacheManager shutdown: {e}")
