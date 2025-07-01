"""
Memory Profiler

Advanced memory profiling and analysis tools for the BERTopic Desktop Application.
Provides detailed memory usage tracking, leak detection, and performance insights.
"""

import logging
import tracemalloc
import psutil
import threading
import time
import gc
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
import pandas as pd
import json

from .memory_manager import MemoryStats, get_memory_monitor

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Detailed memory snapshot at a point in time."""
    
    timestamp: datetime
    process_mb: float
    system_percent: float
    python_objects: int
    python_memory_mb: float
    top_allocations: List[Dict[str, Any]] = field(default_factory=list)
    gc_stats: Dict[str, int] = field(default_factory=dict)
    thread_count: int = 0
    file_descriptors: int = 0
    
    @property
    def memory_pressure_level(self) -> str:
        """Get memory pressure level description."""
        if self.process_mb > 4000 or self.system_percent > 85:
            return "HIGH"
        elif self.process_mb > 2000 or self.system_percent > 70:
            return "MEDIUM"
        else:
            return "LOW"


@dataclass
class MemoryProfileResult:
    """Results from memory profiling session."""
    
    session_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    memory_leak_detected: bool = False
    leak_rate_mb_per_hour: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def average_memory_mb(self) -> float:
        """Calculate average memory usage."""
        if not self.snapshots:
            return 0.0
        return sum(s.process_mb for s in self.snapshots) / len(self.snapshots)
    
    @property
    def memory_growth_mb(self) -> float:
        """Calculate total memory growth."""
        if len(self.snapshots) < 2:
            return 0.0
        return self.snapshots[-1].process_mb - self.snapshots[0].process_mb


class MemoryProfiler:
    """Advanced memory profiler with leak detection and analysis."""
    
    def __init__(self, snapshot_interval: float = 30.0, enable_tracemalloc: bool = True):
        """Initialize the memory profiler.
        
        Args:
            snapshot_interval: Seconds between snapshots
            enable_tracemalloc: Whether to enable Python object tracking
        """
        self.snapshot_interval = snapshot_interval
        self.enable_tracemalloc = enable_tracemalloc
        self.process = psutil.Process()
        self.memory_monitor = get_memory_monitor()
        
        # Profiling state
        self.is_profiling = False
        self.current_session: Optional[str] = None
        self.snapshots: List[MemorySnapshot] = []
        self.start_time: Optional[datetime] = None
        
        # Threading
        self.profiler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Analysis settings
        self.leak_detection_threshold_mb = 50.0  # Memory increase threshold
        self.leak_detection_min_duration = timedelta(minutes=10)  # Minimum duration for leak detection
        
        logger.info("Memory profiler initialized")
    
    def start_profiling(self, session_name: str = None) -> str:
        """Start memory profiling session.
        
        Args:
            session_name: Name for this profiling session
            
        Returns:
            Session ID
        """
        if self.is_profiling:
            logger.warning("Profiling already in progress")
            return self.current_session
        
        session_id = session_name or f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = session_id
        self.snapshots.clear()
        self.start_time = datetime.now()
        
        # Enable tracemalloc if requested
        if self.enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()
            logger.info("Started Python object tracing")
        
        # Start profiling thread
        self.is_profiling = True
        self.stop_event.clear()
        self.profiler_thread = threading.Thread(target=self._profiling_loop, daemon=True)
        self.profiler_thread.start()
        
        logger.info(f"Started memory profiling session: {session_id}")
        return session_id
    
    def stop_profiling(self) -> MemoryProfileResult:
        """Stop profiling and return results.
        
        Returns:
            Profiling results with analysis
        """
        if not self.is_profiling:
            logger.warning("No profiling session in progress")
            return None
        
        # Stop profiling
        self.is_profiling = False
        self.stop_event.set()
        
        if self.profiler_thread:
            self.profiler_thread.join(timeout=5.0)
        
        # Take final snapshot
        final_snapshot = self._take_snapshot()
        if final_snapshot:
            self.snapshots.append(final_snapshot)
        
        # Create results
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        result = MemoryProfileResult(
            session_name=self.current_session,
            start_time=self.start_time,
            end_time=end_time,
            duration_seconds=duration,
            snapshots=self.snapshots.copy(),
            peak_memory_mb=max((s.process_mb for s in self.snapshots), default=0),
        )
        
        # Analyze results
        self._analyze_memory_usage(result)
        
        # Cleanup
        self.current_session = None
        self.snapshots.clear()
        self.start_time = None
        
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()
            logger.info("Stopped Python object tracing")
        
        logger.info(f"Stopped profiling session. Duration: {duration:.1f}s, "
                   f"Peak memory: {result.peak_memory_mb:.1f} MB")
        
        return result
    
    @contextmanager
    def profile_context(self, session_name: str = None):
        """Context manager for profiling a code block.
        
        Args:
            session_name: Name for the profiling session
            
        Yields:
            MemoryProfileResult after the context exits
        """
        session_id = self.start_profiling(session_name)
        try:
            yield session_id
        finally:
            result = self.stop_profiling()
            yield result
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, MemoryProfileResult]:
        """Profile a function call.
        
        Args:
            func: Function to profile
            *args, **kwargs: Function arguments
            
        Returns:
            Tuple of (function result, profiling result)
        """
        session_name = f"func_{func.__name__}_{datetime.now().strftime('%H%M%S')}"
        
        self.start_profiling(session_name)
        try:
            result = func(*args, **kwargs)
        finally:
            profile_result = self.stop_profiling()
        
        return result, profile_result
    
    def _profiling_loop(self):
        """Main profiling loop running in background thread."""
        logger.debug("Memory profiling loop started")
        
        while not self.stop_event.wait(self.snapshot_interval):
            if not self.is_profiling:
                break
            
            try:
                snapshot = self._take_snapshot()
                if snapshot:
                    self.snapshots.append(snapshot)
                    
                    # Log significant changes
                    if len(self.snapshots) > 1:
                        prev_snapshot = self.snapshots[-2]
                        memory_change = snapshot.process_mb - prev_snapshot.process_mb
                        
                        if abs(memory_change) > 50:  # More than 50MB change
                            logger.info(f"Significant memory change: {memory_change:+.1f} MB "
                                      f"(total: {snapshot.process_mb:.1f} MB)")
                
            except Exception as e:
                logger.error(f"Error taking memory snapshot: {e}")
        
        logger.debug("Memory profiling loop ended")
    
    def _take_snapshot(self) -> Optional[MemorySnapshot]:
        """Take a detailed memory snapshot."""
        try:
            # Basic memory info
            memory_info = self.process.memory_info()
            process_mb = memory_info.rss / 1024 / 1024
            system_memory = psutil.virtual_memory()
            system_percent = system_memory.percent
            
            # Thread and file descriptor counts
            try:
                thread_count = self.process.num_threads()
                file_descriptors = self.process.num_fds() if hasattr(self.process, 'num_fds') else 0
            except (psutil.AccessDenied, AttributeError):
                thread_count = 0
                file_descriptors = 0
            
            # Python object tracking
            python_objects = 0
            python_memory_mb = 0.0
            top_allocations = []
            
            if self.enable_tracemalloc and tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                python_memory_mb = current / 1024 / 1024
                
                # Get top allocations
                top_stats = tracemalloc.take_snapshot().statistics('lineno')
                python_objects = len(top_stats)
                
                # Convert top 10 allocations to serializable format
                for stat in top_stats[:10]:
                    top_allocations.append({
                        'size_mb': stat.size / 1024 / 1024,
                        'count': stat.count,
                        'filename': stat.traceback.format()[0] if stat.traceback else 'unknown'
                    })
            
            # Garbage collection stats
            gc_stats = {}
            for i in range(3):
                gc_stats[f'generation_{i}'] = len(gc.get_objects(i))
            
            snapshot = MemorySnapshot(
                timestamp=datetime.now(),
                process_mb=process_mb,
                system_percent=system_percent,
                python_objects=python_objects,
                python_memory_mb=python_memory_mb,
                top_allocations=top_allocations,
                gc_stats=gc_stats,
                thread_count=thread_count,
                file_descriptors=file_descriptors
            )
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error creating memory snapshot: {e}")
            return None
    
    def _analyze_memory_usage(self, result: MemoryProfileResult):
        """Analyze memory usage patterns and detect issues."""
        if len(result.snapshots) < 2:
            return
        
        # Calculate memory growth rate
        first_snapshot = result.snapshots[0]
        last_snapshot = result.snapshots[-1]
        
        memory_growth = last_snapshot.process_mb - first_snapshot.process_mb
        hours = result.duration_seconds / 3600
        
        if hours > 0:
            result.leak_rate_mb_per_hour = memory_growth / hours
        
        # Detect potential memory leaks
        if (result.duration_seconds > self.leak_detection_min_duration.total_seconds() and
            memory_growth > self.leak_detection_threshold_mb):
            result.memory_leak_detected = True
            result.recommendations.append(
                f"Potential memory leak detected: {memory_growth:.1f} MB growth "
                f"over {result.duration_seconds/60:.1f} minutes"
            )
        
        # Check for high memory pressure
        high_pressure_count = sum(1 for s in result.snapshots 
                                 if s.memory_pressure_level == "HIGH")
        
        if high_pressure_count > len(result.snapshots) * 0.3:  # More than 30% of time
            result.recommendations.append(
                "High memory pressure detected frequently. Consider optimizing data structures."
            )
        
        # Check for excessive Python objects
        if result.snapshots and result.snapshots[-1].python_objects > 1000000:  # 1M objects
            result.recommendations.append(
                "Large number of Python objects detected. Consider object pooling or cleanup."
            )
        
        # Check for thread proliferation
        max_threads = max((s.thread_count for s in result.snapshots), default=0)
        if max_threads > 20:
            result.recommendations.append(
                f"High thread count detected ({max_threads}). Consider thread pool optimization."
            )
        
        # Add general recommendations
        if result.peak_memory_mb > 2000:
            result.recommendations.append(
                "High peak memory usage. Consider streaming data processing for large files."
            )
        
        if not result.recommendations:
            result.recommendations.append("No significant memory issues detected.")
    
    def save_profile_report(self, result: MemoryProfileResult, output_path: str):
        """Save detailed profiling report to file.
        
        Args:
            result: Profiling results
            output_path: Path to save report
        """
        report_data = {
            'session_info': {
                'name': result.session_name,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat(),
                'duration_seconds': result.duration_seconds,
                'peak_memory_mb': result.peak_memory_mb,
                'average_memory_mb': result.average_memory_mb,
                'memory_growth_mb': result.memory_growth_mb,
                'leak_detected': result.memory_leak_detected,
                'leak_rate_mb_per_hour': result.leak_rate_mb_per_hour
            },
            'recommendations': result.recommendations,
            'snapshots': []
        }
        
        # Add snapshot data
        for snapshot in result.snapshots:
            snapshot_data = {
                'timestamp': snapshot.timestamp.isoformat(),
                'process_mb': snapshot.process_mb,
                'system_percent': snapshot.system_percent,
                'python_objects': snapshot.python_objects,
                'python_memory_mb': snapshot.python_memory_mb,
                'thread_count': snapshot.thread_count,
                'file_descriptors': snapshot.file_descriptors,
                'memory_pressure': snapshot.memory_pressure_level,
                'gc_stats': snapshot.gc_stats,
                'top_allocations': snapshot.top_allocations[:5]  # Top 5 only
            }
            report_data['snapshots'].append(snapshot_data)
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Saved memory profile report to {output_path}")
    
    def create_memory_chart_data(self, result: MemoryProfileResult) -> pd.DataFrame:
        """Create DataFrame suitable for plotting memory usage over time.
        
        Args:
            result: Profiling results
            
        Returns:
            DataFrame with time series data
        """
        if not result.snapshots:
            return pd.DataFrame()
        
        data = []
        for snapshot in result.snapshots:
            data.append({
                'timestamp': snapshot.timestamp,
                'process_memory_mb': snapshot.process_mb,
                'system_memory_percent': snapshot.system_percent,
                'python_memory_mb': snapshot.python_memory_mb,
                'thread_count': snapshot.thread_count,
                'python_objects': snapshot.python_objects,
                'memory_pressure': snapshot.memory_pressure_level
            })
        
        df = pd.DataFrame(data)
        df['elapsed_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
        
        return df


class QuickMemoryProfiler:
    """Lightweight memory profiler for quick checks."""
    
    @staticmethod
    def quick_profile(func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Quick function profiling with minimal overhead.
        
        Args:
            func: Function to profile
            *args, **kwargs: Function arguments
            
        Returns:
            Simple profiling results
        """
        # Before execution
        before_stats = get_memory_monitor().get_current_stats()
        start_time = datetime.now()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # After execution
        end_time = datetime.now()
        after_stats = get_memory_monitor().get_current_stats()
        
        duration = (end_time - start_time).total_seconds()
        memory_change = after_stats.process_mb - before_stats.process_mb
        
        return {
            'function_name': func.__name__,
            'duration_seconds': duration,
            'memory_change_mb': memory_change,
            'before_memory_mb': before_stats.process_mb,
            'after_memory_mb': after_stats.process_mb,
            'result': result
        }
    
    @staticmethod
    @contextmanager
    def quick_context(name: str = "operation"):
        """Quick context manager for profiling code blocks.
        
        Args:
            name: Name for the operation
            
        Yields:
            Function to get current stats
        """
        before_stats = get_memory_monitor().get_current_stats()
        start_time = datetime.now()
        
        def get_current_stats():
            current_stats = get_memory_monitor().get_current_stats()
            elapsed = (datetime.now() - start_time).total_seconds()
            memory_change = current_stats.process_mb - before_stats.process_mb
            
            return {
                'name': name,
                'elapsed_seconds': elapsed,
                'memory_change_mb': memory_change,
                'current_memory_mb': current_stats.process_mb
            }
        
        try:
            yield get_current_stats
        finally:
            final_stats = get_current_stats()
            logger.info(f"Quick profile [{name}]: {final_stats['elapsed_seconds']:.2f}s, "
                       f"{final_stats['memory_change_mb']:+.1f} MB")


# Global profiler instance
_global_profiler: Optional[MemoryProfiler] = None


def get_memory_profiler() -> MemoryProfiler:
    """Get global memory profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = MemoryProfiler()
    return _global_profiler


def profile_function(func: Callable):
    """Decorator for automatic function profiling.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        profiler = get_memory_profiler()
        result, profile_result = profiler.profile_function(func, *args, **kwargs)
        
        # Log summary
        logger.info(f"Function {func.__name__} profiled: "
                   f"{profile_result.duration_seconds:.2f}s, "
                   f"peak {profile_result.peak_memory_mb:.1f} MB")
        
        return result
    
    return wrapper