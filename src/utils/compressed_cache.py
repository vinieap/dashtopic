"""
Compressed Cache System

Memory-efficient caching with compression, tiered storage, and intelligent cleanup.
Reduces memory footprint while maintaining fast access to cached data.
"""

import logging
import pickle
import gzip
import lz4.frame
import zstd
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List, Protocol
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import weakref
import hashlib
import json

from .memory_manager import MemoryOptimizer, get_memory_monitor, LRUCache

logger = logging.getLogger(__name__)


class CompressionMethod(Enum):
    """Available compression methods."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    NUMPY_COMPRESSED = "numpy_compressed"


@dataclass
class CacheEntry:
    """Represents a cached entry with metadata."""
    
    key: str
    size_bytes: int
    compressed_size_bytes: int
    compression_method: CompressionMethod
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    created: datetime = field(default_factory=datetime.now)
    data_type: str = "unknown"
    ttl_seconds: Optional[int] = None
    
    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self.size_bytes == 0:
            return 1.0
        return self.compressed_size_bytes / self.size_bytes
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        
        expiry_time = self.created + timedelta(seconds=self.ttl_seconds)
        return datetime.now() > expiry_time
    
    @property
    def age_hours(self) -> float:
        """Get age in hours."""
        return (datetime.now() - self.created).total_seconds() / 3600


class Compressor(Protocol):
    """Protocol for compression implementations."""
    
    def compress(self, data: bytes) -> bytes:
        """Compress data."""
        ...
    
    def decompress(self, compressed_data: bytes) -> bytes:
        """Decompress data."""
        ...


class GZipCompressor:
    """GZIP compression implementation."""
    
    def __init__(self, compression_level: int = 6):
        self.compression_level = compression_level
    
    def compress(self, data: bytes) -> bytes:
        return gzip.compress(data, compresslevel=self.compression_level)
    
    def decompress(self, compressed_data: bytes) -> bytes:
        return gzip.decompress(compressed_data)


class LZ4Compressor:
    """LZ4 compression implementation."""
    
    def __init__(self, compression_level: int = 0):
        self.compression_level = compression_level
    
    def compress(self, data: bytes) -> bytes:
        return lz4.frame.compress(data, compression_level=self.compression_level)
    
    def decompress(self, compressed_data: bytes) -> bytes:
        return lz4.frame.decompress(compressed_data)


class ZstdCompressor:
    """Zstandard compression implementation."""
    
    def __init__(self, compression_level: int = 3):
        self.compression_level = compression_level
    
    def compress(self, data: bytes) -> bytes:
        return zstd.compress(data, level=self.compression_level)
    
    def decompress(self, compressed_data: bytes) -> bytes:
        return zstd.decompress(compressed_data)


class NumpyCompressor:
    """Specialized compressor for numpy arrays."""
    
    def __init__(self, compression_level: int = 6):
        self.compression_level = compression_level
    
    def compress(self, data: bytes) -> bytes:
        # For numpy arrays, use specialized compression
        try:
            # Deserialize to check if it's a numpy array
            obj = pickle.loads(data)
            if isinstance(obj, np.ndarray):
                # Use numpy's compressed format
                import io
                buffer = io.BytesIO()
                np.savez_compressed(buffer, array=obj)
                compressed = buffer.getvalue()
                
                # Add a header to identify this as numpy compressed
                header = b'NUMPY_COMPRESSED:'
                return header + compressed
            else:
                # Fall back to gzip for non-numpy data
                return gzip.compress(data, compresslevel=self.compression_level)
        except:
            # Fall back to gzip if anything goes wrong
            return gzip.compress(data, compresslevel=self.compression_level)
    
    def decompress(self, compressed_data: bytes) -> bytes:
        header = b'NUMPY_COMPRESSED:'
        if compressed_data.startswith(header):
            # This is numpy compressed data
            numpy_data = compressed_data[len(header):]
            
            import io
            buffer = io.BytesIO(numpy_data)
            loaded = np.load(buffer)
            array = loaded['array']
            
            # Serialize back to bytes
            return pickle.dumps(array)
        else:
            # Fall back to gzip decompression
            return gzip.decompress(compressed_data)


class CompressedCache:
    """Advanced compressed cache with intelligent storage management."""
    
    def __init__(
        self,
        max_memory_mb: float = 500.0,
        default_compression: CompressionMethod = CompressionMethod.LZ4,
        enable_disk_cache: bool = True,
        disk_cache_dir: Optional[str] = None,
        auto_optimize: bool = True
    ):
        """Initialize compressed cache.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            default_compression: Default compression method
            enable_disk_cache: Whether to enable disk caching
            disk_cache_dir: Directory for disk cache
            auto_optimize: Enable automatic optimization
        """
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.default_compression = default_compression
        self.enable_disk_cache = enable_disk_cache
        self.auto_optimize = auto_optimize
        
        # Storage
        self._memory_cache: Dict[str, bytes] = {}
        self._cache_metadata: Dict[str, CacheEntry] = {}
        self._access_order = LRUCache(max_size=10000)  # Track access order
        
        # Disk cache
        self.disk_cache_dir = None
        if enable_disk_cache:
            cache_dir = disk_cache_dir or (Path.home() / ".bertopic_app" / "compressed_cache")
            self.disk_cache_dir = Path(cache_dir)
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Compression engines
        self._compressors: Dict[CompressionMethod, Compressor] = {
            CompressionMethod.GZIP: GZipCompressor(),
            CompressionMethod.LZ4: LZ4Compressor(),
            CompressionMethod.ZSTD: ZstdCompressor(),
            CompressionMethod.NUMPY_COMPRESSED: NumpyCompressor(),
        }
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'compressions': 0,
            'decompressions': 0,
            'disk_writes': 0,
            'disk_reads': 0,
            'total_compressed_bytes': 0,
            'total_uncompressed_bytes': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background cleanup
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        
        if auto_optimize:
            self._start_background_cleanup()
        
        logger.info(f"Compressed cache initialized: {max_memory_mb} MB limit, "
                   f"compression: {default_compression.value}")
    
    def put(
        self,
        key: str,
        value: Any,
        compression: Optional[CompressionMethod] = None,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Store value in cache with compression.
        
        Args:
            key: Cache key
            value: Value to cache
            compression: Compression method (uses default if None)
            ttl_seconds: Time to live in seconds
            
        Returns:
            True if stored successfully
        """
        with self._lock:
            try:
                # Serialize value
                serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                original_size = len(serialized)
                
                # Choose compression method
                if compression is None:
                    compression = self._choose_optimal_compression(value, serialized)
                
                # Compress data
                if compression == CompressionMethod.NONE:
                    compressed_data = serialized
                    compressed_size = original_size
                else:
                    compressor = self._compressors.get(compression)
                    if compressor:
                        compressed_data = compressor.compress(serialized)
                        compressed_size = len(compressed_data)
                        self.stats['compressions'] += 1
                    else:
                        logger.warning(f"Unknown compression method: {compression}")
                        compressed_data = serialized
                        compressed_size = original_size
                        compression = CompressionMethod.NONE
                
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    size_bytes=original_size,
                    compressed_size_bytes=compressed_size,
                    compression_method=compression,
                    data_type=type(value).__name__,
                    ttl_seconds=ttl_seconds
                )
                
                # Check if we need to make space
                if compressed_size > self.max_memory_bytes:
                    logger.warning(f"Item too large for cache: {compressed_size / 1024**2:.1f} MB")
                    return False
                
                # Ensure we have enough space
                self._ensure_space(compressed_size)
                
                # Store in memory or disk
                if self._should_store_in_memory(entry):
                    self._memory_cache[key] = compressed_data
                    logger.debug(f"Stored in memory: {key} ({compressed_size} bytes)")
                elif self.enable_disk_cache:
                    self._store_to_disk(key, compressed_data)
                    logger.debug(f"Stored to disk: {key} ({compressed_size} bytes)")
                else:
                    logger.warning(f"Cannot store item: {key} (no storage available)")
                    return False
                
                # Update metadata and stats
                self._cache_metadata[key] = entry
                self._access_order.put(key, datetime.now())
                
                self.stats['total_compressed_bytes'] += compressed_size
                self.stats['total_uncompressed_bytes'] += original_size
                
                logger.debug(f"Cached {key}: {original_size} -> {compressed_size} bytes "
                           f"({entry.compression_ratio:.2f} ratio)")
                
                return True
                
            except Exception as e:
                logger.error(f"Error caching item {key}: {e}")
                return False
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self._lock:
            try:
                # Check if key exists and is not expired
                if key not in self._cache_metadata:
                    self.stats['misses'] += 1
                    return None
                
                entry = self._cache_metadata[key]
                if entry.is_expired:
                    self._remove_entry(key)
                    self.stats['misses'] += 1
                    return None
                
                # Try to get from memory first
                compressed_data = self._memory_cache.get(key)
                
                # If not in memory, try disk
                if compressed_data is None and self.enable_disk_cache:
                    compressed_data = self._load_from_disk(key)
                    if compressed_data:
                        self.stats['disk_reads'] += 1
                        # Promote to memory if there's space
                        if self._get_memory_usage() + len(compressed_data) <= self.max_memory_bytes:
                            self._memory_cache[key] = compressed_data
                
                if compressed_data is None:
                    logger.warning(f"Cache entry metadata exists but data not found: {key}")
                    self._remove_entry(key)
                    self.stats['misses'] += 1
                    return None
                
                # Decompress data
                if entry.compression_method == CompressionMethod.NONE:
                    serialized_data = compressed_data
                else:
                    compressor = self._compressors.get(entry.compression_method)
                    if compressor:
                        serialized_data = compressor.decompress(compressed_data)
                        self.stats['decompressions'] += 1
                    else:
                        logger.error(f"Unknown compression method: {entry.compression_method}")
                        self.stats['misses'] += 1
                        return None
                
                # Deserialize value
                value = pickle.loads(serialized_data)
                
                # Update access statistics
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                self._access_order.put(key, datetime.now())
                
                self.stats['hits'] += 1
                
                logger.debug(f"Cache hit: {key} (accessed {entry.access_count} times)")
                return value
                
            except Exception as e:
                logger.error(f"Error retrieving cached item {key}: {e}")
                self.stats['misses'] += 1
                return None
    
    def contains(self, key: str) -> bool:
        """Check if key exists in cache and is not expired.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists and is valid
        """
        with self._lock:
            if key not in self._cache_metadata:
                return False
            
            entry = self._cache_metadata[key]
            if entry.is_expired:
                self._remove_entry(key)
                return False
            
            return True
    
    def remove(self, key: str) -> bool:
        """Remove item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if item was removed
        """
        with self._lock:
            return self._remove_entry(key)
    
    def clear(self):
        """Clear all cached items."""
        with self._lock:
            # Clear memory cache
            self._memory_cache.clear()
            
            # Clear disk cache
            if self.enable_disk_cache and self.disk_cache_dir:
                for cache_file in self.disk_cache_dir.glob("*.cache"):
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.warning(f"Error removing cache file {cache_file}: {e}")
            
            # Clear metadata
            self._cache_metadata.clear()
            self._access_order.clear()
            
            # Reset stats
            for key in self.stats:
                if key.startswith('total_'):
                    self.stats[key] = 0
            
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            hit_rate = 0.0
            total_requests = self.stats['hits'] + self.stats['misses']
            if total_requests > 0:
                hit_rate = self.stats['hits'] / total_requests
            
            memory_usage = self._get_memory_usage()
            
            # Calculate compression efficiency
            compression_ratio = 1.0
            if self.stats['total_uncompressed_bytes'] > 0:
                compression_ratio = (self.stats['total_compressed_bytes'] / 
                                   self.stats['total_uncompressed_bytes'])
            
            return {
                'hit_rate': hit_rate,
                'total_entries': len(self._cache_metadata),
                'memory_entries': len(self._memory_cache),
                'memory_usage_mb': memory_usage / 1024**2,
                'memory_limit_mb': self.max_memory_bytes / 1024**2,
                'memory_utilization': memory_usage / self.max_memory_bytes,
                'compression_ratio': compression_ratio,
                'disk_cache_enabled': self.enable_disk_cache,
                **self.stats
            }
    
    def optimize(self):
        """Run cache optimization."""
        with self._lock:
            logger.info("Running cache optimization")
            
            # Remove expired entries
            expired_keys = [key for key, entry in self._cache_metadata.items() 
                          if entry.is_expired]
            
            for key in expired_keys:
                self._remove_entry(key)
            
            logger.info(f"Removed {len(expired_keys)} expired entries")
            
            # Optimize memory usage
            self._optimize_memory_allocation()
            
            # Run garbage collection
            MemoryOptimizer.force_garbage_collection()
    
    def _choose_optimal_compression(self, value: Any, serialized: bytes) -> CompressionMethod:
        """Choose optimal compression method based on data type and size.
        
        Args:
            value: Original value
            serialized: Serialized bytes
            
        Returns:
            Optimal compression method
        """
        data_size = len(serialized)
        
        # For small data, compression overhead may not be worth it
        if data_size < 1024:  # Less than 1KB
            return CompressionMethod.NONE
        
        # For numpy arrays, use specialized compression
        if isinstance(value, np.ndarray):
            return CompressionMethod.NUMPY_COMPRESSED
        
        # For medium-sized data, use LZ4 (fast)
        if data_size < 10 * 1024 * 1024:  # Less than 10MB
            return CompressionMethod.LZ4
        
        # For large data, use ZSTD (better compression)
        return CompressionMethod.ZSTD
    
    def _should_store_in_memory(self, entry: CacheEntry) -> bool:
        """Determine if entry should be stored in memory.
        
        Args:
            entry: Cache entry
            
        Returns:
            True if should store in memory
        """
        # Always try memory first if there's space
        current_usage = self._get_memory_usage()
        if current_usage + entry.compressed_size_bytes <= self.max_memory_bytes:
            return True
        
        # Don't store very large items in memory
        if entry.compressed_size_bytes > self.max_memory_bytes * 0.1:  # More than 10% of limit
            return False
        
        return False
    
    def _ensure_space(self, required_bytes: int):
        """Ensure there's enough space in cache.
        
        Args:
            required_bytes: Required space in bytes
        """
        current_usage = self._get_memory_usage()
        
        while current_usage + required_bytes > self.max_memory_bytes:
            # Find least recently used item in memory
            lru_key = None
            lru_time = datetime.now()
            
            for key in self._memory_cache.keys():
                if key in self._cache_metadata:
                    entry = self._cache_metadata[key]
                    if entry.last_accessed < lru_time:
                        lru_time = entry.last_accessed
                        lru_key = key
            
            if lru_key is None:
                break
            
            # Move to disk or remove entirely
            if self.enable_disk_cache:
                self._move_to_disk(lru_key)
            else:
                self._remove_entry(lru_key)
                self.stats['evictions'] += 1
            
            current_usage = self._get_memory_usage()
    
    def _move_to_disk(self, key: str):
        """Move entry from memory to disk.
        
        Args:
            key: Cache key
        """
        if key not in self._memory_cache:
            return
        
        compressed_data = self._memory_cache[key]
        
        if self._store_to_disk(key, compressed_data):
            del self._memory_cache[key]
            logger.debug(f"Moved to disk: {key}")
        
    def _store_to_disk(self, key: str, compressed_data: bytes) -> bool:
        """Store compressed data to disk.
        
        Args:
            key: Cache key
            compressed_data: Compressed data
            
        Returns:
            True if stored successfully
        """
        if not self.enable_disk_cache or not self.disk_cache_dir:
            return False
        
        try:
            # Create safe filename
            safe_key = hashlib.md5(key.encode()).hexdigest()
            cache_file = self.disk_cache_dir / f"{safe_key}.cache"
            
            with open(cache_file, 'wb') as f:
                f.write(compressed_data)
            
            self.stats['disk_writes'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error storing to disk: {e}")
            return False
    
    def _load_from_disk(self, key: str) -> Optional[bytes]:
        """Load compressed data from disk.
        
        Args:
            key: Cache key
            
        Returns:
            Compressed data or None if not found
        """
        if not self.enable_disk_cache or not self.disk_cache_dir:
            return None
        
        try:
            safe_key = hashlib.md5(key.encode()).hexdigest()
            cache_file = self.disk_cache_dir / f"{safe_key}.cache"
            
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return f.read()
            
        except Exception as e:
            logger.error(f"Error loading from disk: {e}")
        
        return None
    
    def _remove_entry(self, key: str) -> bool:
        """Remove entry completely.
        
        Args:
            key: Cache key
            
        Returns:
            True if removed
        """
        removed = False
        
        # Remove from memory
        if key in self._memory_cache:
            del self._memory_cache[key]
            removed = True
        
        # Remove from disk
        if self.enable_disk_cache and self.disk_cache_dir:
            try:
                safe_key = hashlib.md5(key.encode()).hexdigest()
                cache_file = self.disk_cache_dir / f"{safe_key}.cache"
                if cache_file.exists():
                    cache_file.unlink()
                    removed = True
            except Exception as e:
                logger.warning(f"Error removing disk cache file: {e}")
        
        # Remove metadata
        if key in self._cache_metadata:
            del self._cache_metadata[key]
            removed = True
        
        return removed
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes.
        
        Returns:
            Memory usage in bytes
        """
        return sum(len(data) for data in self._memory_cache.values())
    
    def _optimize_memory_allocation(self):
        """Optimize memory allocation by promoting frequently accessed items."""
        if not self.enable_disk_cache:
            return
        
        # Get available memory space
        available_memory = self.max_memory_bytes - self._get_memory_usage()
        
        if available_memory <= 0:
            return
        
        # Find disk-cached items that should be promoted to memory
        candidates = []
        for key, entry in self._cache_metadata.items():
            if key not in self._memory_cache and entry.access_count > 1:
                candidates.append((key, entry))
        
        # Sort by access frequency
        candidates.sort(key=lambda x: x[1].access_count, reverse=True)
        
        # Promote items to memory
        for key, entry in candidates:
            if entry.compressed_size_bytes <= available_memory:
                compressed_data = self._load_from_disk(key)
                if compressed_data:
                    self._memory_cache[key] = compressed_data
                    available_memory -= entry.compressed_size_bytes
                    logger.debug(f"Promoted to memory: {key}")
            
            if available_memory <= 0:
                break
    
    def _start_background_cleanup(self):
        """Start background cleanup thread."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
        
        self._stop_cleanup.clear()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        logger.debug("Started background cache cleanup")
    
    def _cleanup_loop(self):
        """Background cleanup loop."""
        while not self._stop_cleanup.wait(300):  # Run every 5 minutes
            try:
                self.optimize()
            except Exception as e:
                logger.error(f"Error in background cleanup: {e}")
    
    def stop_background_cleanup(self):
        """Stop background cleanup thread."""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=1.0)
    
    def __del__(self):
        """Cleanup when cache is destroyed."""
        try:
            self.stop_background_cleanup()
        except:
            pass


# Global compressed cache instance
_global_compressed_cache: Optional[CompressedCache] = None


def get_compressed_cache() -> CompressedCache:
    """Get global compressed cache instance."""
    global _global_compressed_cache
    if _global_compressed_cache is None:
        _global_compressed_cache = CompressedCache()
    return _global_compressed_cache