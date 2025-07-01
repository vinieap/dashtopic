"""
Memory Management Utilities

This module provides memory monitoring, optimization, and management utilities
for the BERTopic Desktop Application.

Features:
- Memory usage monitoring
- Memory pressure detection  
- Garbage collection optimization
- Memory-efficient data loading
- Resource cleanup utilities
"""

import gc
import logging
import psutil
import os
import threading
import weakref
from typing import Dict, Any, Optional, List, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    
    process_mb: float  # Process memory in MB
    process_percent: float  # Process memory as percentage of total
    system_percent: float  # System memory usage percentage  
    available_mb: float  # Available system memory in MB
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_high_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        return self.system_percent > 85.0 or self.process_mb > 4000  # 4GB
    
    @property
    def is_medium_pressure(self) -> bool:
        """Check if system has medium memory pressure."""
        return self.system_percent > 70.0 or self.process_mb > 2000  # 2GB


class MemoryMonitor:
    """Monitor and track memory usage patterns."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.process = psutil.Process(os.getpid())
        self.history: List[MemoryStats] = []
        self.max_history = 100
        self.callbacks: List[Callable[[MemoryStats], None]] = []
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
    def get_current_stats(self) -> MemoryStats:
        """Get current memory usage statistics."""
        try:
            # Get process memory info
            memory_info = self.process.memory_info()
            process_mb = memory_info.rss / 1024 / 1024
            process_percent = self.process.memory_percent()
            
            # Get system memory info
            system_memory = psutil.virtual_memory()
            system_percent = system_memory.percent
            available_mb = system_memory.available / 1024 / 1024
            
            stats = MemoryStats(
                process_mb=process_mb,
                process_percent=process_percent,
                system_percent=system_percent,
                available_mb=available_mb
            )
            
            # Add to history
            self.history.append(stats)
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback(stats)
                except Exception as e:
                    logger.error(f"Error in memory monitor callback: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return MemoryStats(0, 0, 0, 0)
    
    def add_callback(self, callback: Callable[[MemoryStats], None]):
        """Add a callback to be called when memory stats are updated."""
        self.callbacks.append(callback)
    
    def start_monitoring(self):
        """Start background memory monitoring."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self._stop_monitoring.wait(self.check_interval):
            self.get_current_stats()
    
    def get_peak_usage(self) -> Optional[MemoryStats]:
        """Get peak memory usage from history."""
        if not self.history:
            return None
        return max(self.history, key=lambda s: s.process_mb)
    
    def get_average_usage(self) -> Optional[float]:
        """Get average memory usage from history."""
        if not self.history:
            return None
        return sum(s.process_mb for s in self.history) / len(self.history)


class MemoryOptimizer:
    """Utilities for memory optimization and cleanup."""
    
    @staticmethod
    def force_garbage_collection() -> Dict[str, int]:
        """Force comprehensive garbage collection."""
        # Clear matplotlib figure cache
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except ImportError:
            pass
        
        # Collect garbage multiple times for better cleanup
        collected = {}
        for generation in range(3):
            count = gc.collect(generation)
            collected[f"generation_{generation}"] = count
        
        # Force full collection
        total_collected = gc.collect()
        collected["total"] = total_collected
        
        logger.info(f"Garbage collection completed: {collected}")
        return collected
    
    @staticmethod
    def optimize_pandas_memory(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize pandas DataFrame memory usage."""
        original_memory = df.memory_usage(deep=True).sum()
        
        # Optimize numeric columns
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                # Optimize numeric types
                if str(col_type).startswith('int'):
                    # Try smaller integer types
                    if df[col].min() >= -128 and df[col].max() <= 127:
                        df[col] = df[col].astype('int8')
                    elif df[col].min() >= -32768 and df[col].max() <= 32767:
                        df[col] = df[col].astype('int16')
                    elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                        df[col] = df[col].astype('int32')
                
                elif str(col_type).startswith('float'):
                    # Try float32 if precision allows
                    if df[col].dtype == 'float64':
                        df[col] = pd.to_numeric(df[col], downcast='float')
            
            else:
                # Optimize object columns
                if df[col].nunique() / len(df) < 0.5:  # Low cardinality
                    df[col] = df[col].astype('category')
        
        optimized_memory = df.memory_usage(deep=True).sum()
        reduction = (original_memory - optimized_memory) / original_memory * 100
        
        logger.info(f"DataFrame memory optimized: {reduction:.1f}% reduction "
                   f"({original_memory / 1024**2:.1f} MB -> {optimized_memory / 1024**2:.1f} MB)")
        
        return df
    
    @staticmethod
    def optimize_numpy_array(arr: np.ndarray, target_dtype: Optional[str] = None) -> np.ndarray:
        """Optimize numpy array memory usage."""
        original_size = arr.nbytes
        
        if target_dtype:
            optimized = arr.astype(target_dtype)
        else:
            # Auto-optimize based on data range
            if arr.dtype == np.float64:
                # Try float32 if data range allows
                if np.allclose(arr, arr.astype(np.float32), rtol=1e-6):
                    optimized = arr.astype(np.float32)
                else:
                    optimized = arr
            elif arr.dtype == np.int64:
                # Try smaller integer types
                if arr.min() >= -128 and arr.max() <= 127:
                    optimized = arr.astype(np.int8)
                elif arr.min() >= -32768 and arr.max() <= 32767:
                    optimized = arr.astype(np.int16)
                elif arr.min() >= -2147483648 and arr.max() <= 2147483647:
                    optimized = arr.astype(np.int32)
                else:
                    optimized = arr
            else:
                optimized = arr
        
        optimized_size = optimized.nbytes
        reduction = (original_size - optimized_size) / original_size * 100
        
        if reduction > 0:
            logger.info(f"Array memory optimized: {reduction:.1f}% reduction "
                       f"({original_size / 1024**2:.1f} MB -> {optimized_size / 1024**2:.1f} MB)")
        
        return optimized


class LRUCache(Generic[T]):
    """Memory-efficient LRU cache with automatic cleanup."""
    
    def __init__(self, max_size: int, cleanup_callback: Optional[Callable[[T], None]] = None):
        self.max_size = max_size
        self.cleanup_callback = cleanup_callback
        self._cache: Dict[str, T] = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[T]:
        """Get item from cache."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
            return None
    
    def put(self, key: str, value: T) -> None:
        """Put item in cache."""
        with self._lock:
            if key in self._cache:
                # Update existing item
                self._access_order.remove(key)
            elif len(self._cache) >= self.max_size:
                # Remove least recently used item
                lru_key = self._access_order.pop(0)
                lru_value = self._cache.pop(lru_key)
                
                # Call cleanup callback
                if self.cleanup_callback:
                    try:
                        self.cleanup_callback(lru_value)
                    except Exception as e:
                        logger.error(f"Error in cache cleanup callback: {e}")
            
            self._cache[key] = value
            self._access_order.append(key)
    
    def clear(self) -> None:
        """Clear all items from cache."""
        with self._lock:
            if self.cleanup_callback:
                for value in self._cache.values():
                    try:
                        self.cleanup_callback(value)
                    except Exception as e:
                        logger.error(f"Error in cache cleanup callback: {e}")
            
            self._cache.clear()
            self._access_order.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())


class WeakValueCache:
    """Cache that automatically removes items when they're garbage collected."""
    
    def __init__(self):
        self._cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            return self._cache.get(key)
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self._lock:
            self._cache[key] = value
    
    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class MemoryPressureManager:
    """Manage application behavior under memory pressure."""
    
    def __init__(self, memory_monitor: MemoryMonitor):
        self.memory_monitor = memory_monitor
        self.pressure_callbacks: Dict[str, List[Callable[[], None]]] = {
            'medium': [],
            'high': []
        }
        self.memory_monitor.add_callback(self._check_pressure)
        self._last_cleanup = datetime.now()
        self._cleanup_interval = timedelta(minutes=5)
    
    def register_pressure_callback(self, level: str, callback: Callable[[], None]):
        """Register callback for memory pressure events."""
        if level in self.pressure_callbacks:
            self.pressure_callbacks[level].append(callback)
    
    def _check_pressure(self, stats: MemoryStats):
        """Check memory pressure and trigger callbacks."""
        now = datetime.now()
        
        if stats.is_high_pressure:
            logger.warning(f"High memory pressure detected: {stats.process_mb:.1f} MB "
                          f"({stats.system_percent:.1f}% system)")
            self._trigger_callbacks('high')
            
            # Force cleanup
            if now - self._last_cleanup > self._cleanup_interval:
                self._emergency_cleanup()
                self._last_cleanup = now
        
        elif stats.is_medium_pressure:
            logger.info(f"Medium memory pressure detected: {stats.process_mb:.1f} MB "
                       f"({stats.system_percent:.1f}% system)")
            self._trigger_callbacks('medium')
    
    def _trigger_callbacks(self, level: str):
        """Trigger pressure callbacks for given level."""
        for callback in self.pressure_callbacks[level]:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in memory pressure callback: {e}")
    
    def _emergency_cleanup(self):
        """Perform emergency memory cleanup."""
        logger.warning("Performing emergency memory cleanup")
        
        # Force garbage collection
        MemoryOptimizer.force_garbage_collection()
        
        # Clear caches (would be implemented by cache services)
        self._trigger_callbacks('high')


# Global memory monitor instance
_global_memory_monitor: Optional[MemoryMonitor] = None


def get_memory_monitor() -> MemoryMonitor:
    """Get global memory monitor instance."""
    global _global_memory_monitor
    if _global_memory_monitor is None:
        _global_memory_monitor = MemoryMonitor()
    return _global_memory_monitor


def start_memory_monitoring():
    """Start global memory monitoring."""
    monitor = get_memory_monitor()
    monitor.start_monitoring()


def stop_memory_monitoring():
    """Stop global memory monitoring."""
    monitor = get_memory_monitor()
    monitor.stop_monitoring()


def get_memory_stats() -> MemoryStats:
    """Get current memory statistics."""
    monitor = get_memory_monitor()
    return monitor.get_current_stats()