# Memory Optimization Implementation Guide

## 🚀 Quick Start Implementation

This guide provides step-by-step instructions for implementing the memory optimizations in your BERTopic Desktop Application.

## Phase 1: Basic Integration (Day 1)

### 1. Update File Loading

**Replace existing file loading with streaming:**

```python
# OLD - loads entire file into memory
def load_data_file(file_path):
    file_service = FileIOService()
    df, metadata = file_service.load_file(file_path)
    return df, metadata

# NEW - uses streaming and samples
def load_data_file(file_path, use_sample=True):
    file_service = FileIOService(enable_streaming=True)
    
    if use_sample:
        # Load sample for preview/analysis
        df, metadata = file_service.load_file_sample(file_path, sample_size=1000)
        return df, metadata
    else:
        # Stream for processing
        chunks = []
        metadata = None
        for chunk, chunk_metadata in file_service.load_file_streaming(file_path):
            if metadata is None:
                metadata = chunk_metadata
            chunks.append(chunk)
        
        # Combine chunks if needed (or process individually)
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            return df, metadata
        else:
            return None, None
```

### 2. Add Memory Monitoring

**Add to application startup:**

```python
# In your main application initialization
from src.utils.memory_manager import start_memory_monitoring

def initialize_application():
    # Start memory monitoring
    start_memory_monitoring()
    
    # Your existing initialization code
    setup_ui()
    load_config()
    # etc.
```

**Add to data processing functions:**

```python
from src.utils.memory_manager import get_memory_stats, MemoryOptimizer

def process_data(df):
    # Check memory before processing
    before_stats = get_memory_stats()
    print(f"Memory before: {before_stats.process_mb:.1f} MB")
    
    # Optimize DataFrame
    df_optimized = MemoryOptimizer.optimize_pandas_memory(df)
    
    # Your processing logic
    result = perform_analysis(df_optimized)
    
    # Check memory after and cleanup if needed
    after_stats = get_memory_stats()
    if after_stats.is_high_pressure:
        MemoryOptimizer.force_garbage_collection()
    
    return result
```

### 3. Enable Compressed Caching

**Update cache service initialization:**

```python
# OLD
cache_service = CacheService()

# NEW
from src.utils.compressed_cache import CompressionMethod

cache_service = CacheService(
    enable_compression=True,
    compression_method=CompressionMethod.LZ4,
    max_cache_size_gb=3.0  # Adjust based on your system
)
```

**Update embedding caching:**

```python
# OLD
success = cache_service.save_embeddings(cache_key, embeddings, texts, model_info, data_hash)
embeddings, texts = cache_service.get_cached_embeddings(cache_key)

# NEW
success = cache_service.save_embeddings_compressed(cache_key, embeddings, texts, model_info, data_hash)
embeddings, texts = cache_service.get_cached_embeddings_fast(cache_key)
```

## Phase 2: Advanced Features (Week 1)

### 1. Add Memory Profiling

**For debugging memory issues:**

```python
from src.utils.memory_profiler import get_memory_profiler

def debug_memory_intensive_function():
    profiler = get_memory_profiler()
    
    # Profile the operation
    profiler.start_profiling("embedding_generation")
    
    # Your memory-intensive code
    embeddings = generate_embeddings(texts)
    
    # Get results
    result = profiler.stop_profiling()
    
    # Log results
    print(f"Peak memory: {result.peak_memory_mb:.1f} MB")
    print(f"Duration: {result.duration_seconds:.1f} seconds")
    if result.memory_leak_detected:
        print("⚠️ Potential memory leak detected!")
        print("Recommendations:", result.recommendations)
    
    return embeddings
```

**For automatic function profiling:**

```python
from src.utils.memory_profiler import profile_function

@profile_function
def generate_embeddings(texts):
    # Function is automatically profiled
    # Results logged automatically
    return model.encode(texts)
```

### 2. Implement Chunked Processing

**For large datasets:**

```python
from src.services.file_io_service import FileIOService

def process_large_dataset(file_path):
    file_service = FileIOService(enable_streaming=True, streaming_chunk_size=5000)
    
    results = []
    
    def process_chunk(chunk):
        # Process individual chunk
        processed = analyze_chunk(chunk)
        return processed
    
    def combine_results(chunk_results):
        # Combine results from all chunks
        return pd.concat(chunk_results, ignore_index=True)
    
    # Process entire file in chunks
    final_result = file_service.process_file_in_chunks(
        file_path,
        process_chunk,
        combine_results
    )
    
    return final_result
```

### 3. Add Memory Pressure Handling

**Automatic optimization under pressure:**

```python
from src.utils.memory_manager import MemoryPressureManager, get_memory_monitor

def setup_memory_pressure_handling():
    monitor = get_memory_monitor()
    pressure_manager = MemoryPressureManager(monitor)
    
    # Define what to do under medium pressure
    def handle_medium_pressure():
        print("⚠️ Medium memory pressure - optimizing...")
        # Force DataFrame optimization
        global current_dataframes
        for df_name, df in current_dataframes.items():
            current_dataframes[df_name] = MemoryOptimizer.optimize_pandas_memory(df)
        
        # Optimize cache
        cache_service.optimize_cache()
    
    # Define what to do under high pressure
    def handle_high_pressure():
        print("🚨 High memory pressure - emergency cleanup!")
        # Clear non-essential caches
        cache_service.clear_all_caches()
        
        # Force garbage collection
        MemoryOptimizer.force_garbage_collection()
        
        # Switch to disk-only mode temporarily
        switch_to_minimal_memory_mode()
    
    # Register handlers
    pressure_manager.register_pressure_callback('medium', handle_medium_pressure)
    pressure_manager.register_pressure_callback('high', handle_high_pressure)
    
    return pressure_manager
```

## Phase 3: UI Integration (Week 2)

### 1. Add Memory Status to UI

**Create memory status widget:**

```python
import customtkinter as ctk
from src.utils.memory_manager import get_memory_stats

class MemoryStatusWidget(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.memory_label = ctk.CTkLabel(self, text="Memory: 0 MB")
        self.memory_label.pack(pady=5)
        
        self.pressure_label = ctk.CTkLabel(self, text="Pressure: LOW")
        self.pressure_label.pack(pady=5)
        
        # Update every 5 seconds
        self.update_memory_status()
    
    def update_memory_status(self):
        stats = get_memory_stats()
        
        # Update memory usage
        self.memory_label.configure(text=f"Memory: {stats.process_mb:.0f} MB")
        
        # Update pressure indicator with colors
        if stats.is_high_pressure:
            self.pressure_label.configure(text="Pressure: HIGH", text_color="red")
        elif stats.is_medium_pressure:
            self.pressure_label.configure(text="Pressure: MEDIUM", text_color="orange")
        else:
            self.pressure_label.configure(text="Pressure: LOW", text_color="green")
        
        # Schedule next update
        self.after(5000, self.update_memory_status)

# Add to your main window
memory_widget = MemoryStatusWidget(main_window)
memory_widget.pack(side="bottom", fill="x")
```

### 2. Add Cache Statistics

**Cache stats widget:**

```python
class CacheStatsWidget(ctk.CTkFrame):
    def __init__(self, parent, cache_service):
        super().__init__(parent)
        self.cache_service = cache_service
        
        self.stats_text = ctk.CTkTextbox(self, height=100)
        self.stats_text.pack(fill="both", expand=True)
        
        self.update_button = ctk.CTkButton(self, text="Update Stats", command=self.update_stats)
        self.update_button.pack(pady=5)
        
        self.optimize_button = ctk.CTkButton(self, text="Optimize Cache", command=self.optimize_cache)
        self.optimize_button.pack(pady=5)
        
        self.update_stats()
    
    def update_stats(self):
        stats = self.cache_service.get_enhanced_cache_stats()
        
        stats_text = f"""Cache Statistics:
Regular Cache: {stats['total_files']} files, {stats['total_size_mb']:.1f} MB
Compressed Cache: {stats['compressed_cache']['total_entries']} entries
Memory Usage: {stats['compressed_cache']['memory_usage_mb']:.1f} MB
Hit Rate: {stats['compressed_cache']['hit_rate']:.1%}
Compression Ratio: {stats['compressed_cache']['compression_ratio']:.2f}
Memory Saved: {stats['total_cache_efficiency']['compression_savings_mb']:.1f} MB"""
        
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", stats_text)
    
    def optimize_cache(self):
        self.cache_service.optimize_cache()
        self.update_stats()
        print("✅ Cache optimization completed")
```

### 3. Smart File Loading Dialog

**File loading with memory awareness:**

```python
class SmartFileDialog(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Load Data File")
        
        # File selection
        self.file_var = ctk.StringVar()
        file_frame = ctk.CTkFrame(self)
        file_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(file_frame, text="File:").pack(side="left")
        ctk.CTkEntry(file_frame, textvariable=self.file_var).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(file_frame, text="Browse", command=self.browse_file).pack(side="right")
        
        # Loading options
        options_frame = ctk.CTkFrame(self)
        options_frame.pack(fill="x", padx=10, pady=5)
        
        self.load_mode = ctk.StringVar(value="sample")
        ctk.CTkRadioButton(options_frame, text="Load Sample (Fast, Low Memory)", 
                          variable=self.load_mode, value="sample").pack(anchor="w")
        ctk.CTkRadioButton(options_frame, text="Stream Processing (Medium Memory)", 
                          variable=self.load_mode, value="stream").pack(anchor="w")
        ctk.CTkRadioButton(options_frame, text="Full Load (High Memory)", 
                          variable=self.load_mode, value="full").pack(anchor="w")
        
        # Memory status
        self.memory_label = ctk.CTkLabel(self, text="")
        self.memory_label.pack(pady=5)
        
        # Buttons
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkButton(button_frame, text="Load", command=self.load_file).pack(side="right", padx=5)
        ctk.CTkButton(button_frame, text="Cancel", command=self.destroy).pack(side="right")
        
        self.update_memory_status()
    
    def browse_file(self):
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if filename:
            self.file_var.set(filename)
            self.estimate_memory_usage()
    
    def estimate_memory_usage(self):
        file_path = self.file_var.get()
        if not file_path:
            return
        
        file_service = FileIOService(enable_streaming=True)
        file_info = file_service.get_file_info_efficient(file_path)
        
        # Estimate memory usage
        estimated_mb = file_info['file_size_mb'] * 3  # Rough estimate
        
        mode = self.load_mode.get()
        if mode == "sample":
            estimated_mb = min(estimated_mb, 50)  # Sample limited to ~50MB
        elif mode == "stream":
            estimated_mb = min(estimated_mb, 200)  # Streaming uses chunks
        
        self.memory_label.configure(text=f"Estimated memory usage: {estimated_mb:.1f} MB")
        
        # Warn if too large
        current_stats = get_memory_stats()
        if current_stats.process_mb + estimated_mb > 3000:  # More than 3GB total
            self.memory_label.configure(text_color="red")
        elif current_stats.process_mb + estimated_mb > 2000:  # More than 2GB total
            self.memory_label.configure(text_color="orange")
        else:
            self.memory_label.configure(text_color="green")
    
    def update_memory_status(self):
        stats = get_memory_stats()
        current_text = self.memory_label.cget("text")
        if not current_text.startswith("Estimated"):
            self.memory_label.configure(text=f"Current memory: {stats.process_mb:.1f} MB")
        
        self.after(2000, self.update_memory_status)
    
    def load_file(self):
        file_path = self.file_var.get()
        mode = self.load_mode.get()
        
        if not file_path:
            return
        
        file_service = FileIOService(enable_streaming=True)
        
        try:
            if mode == "sample":
                df, metadata = file_service.load_file_sample(file_path, sample_size=1000)
                self.result = (df, metadata, "sample")
            elif mode == "stream":
                # For streaming, return generator
                stream_gen = file_service.load_file_streaming(file_path)
                self.result = (stream_gen, None, "stream")
            else:  # full
                df, metadata = file_service.load_file(file_path)
                self.result = (df, metadata, "full")
            
            self.destroy()
            
        except Exception as e:
            print(f"Error loading file: {e}")
```

## Phase 4: Production Optimization (Ongoing)

### 1. Configuration Management

**Create memory config file:**

```python
# config/memory_config.py
from dataclasses import dataclass
from src.utils.compressed_cache import CompressionMethod

@dataclass
class MemoryConfig:
    # Streaming settings
    enable_streaming: bool = True
    streaming_chunk_size: int = 10000
    memory_limit_mb: float = 2000.0
    
    # Cache settings
    enable_compression: bool = True
    compression_method: CompressionMethod = CompressionMethod.LZ4
    max_cache_size_gb: float = 3.0
    
    # Monitoring settings
    enable_memory_monitoring: bool = True
    memory_snapshot_interval: float = 30.0
    enable_memory_profiling: bool = False  # Only for debugging
    
    # Pressure thresholds
    medium_pressure_threshold_mb: float = 2000.0
    high_pressure_threshold_mb: float = 4000.0
    
    @classmethod
    def load_from_file(cls, config_path: str):
        # Load configuration from file
        pass
    
    def save_to_file(self, config_path: str):
        # Save configuration to file
        pass

# Usage
config = MemoryConfig.load_from_file("config/memory.json")
file_service = FileIOService(
    enable_streaming=config.enable_streaming,
    streaming_chunk_size=config.streaming_chunk_size,
    memory_limit_mb=config.memory_limit_mb
)
```

### 2. Automated Testing

**Memory performance tests:**

```python
import pytest
from src.utils.memory_profiler import QuickMemoryProfiler

def test_memory_usage_file_loading():
    """Test that file loading stays within memory limits."""
    with QuickMemoryProfiler.quick_context("file_loading") as get_stats:
        file_service = FileIOService(enable_streaming=True)
        df, metadata = file_service.load_file_sample("test_data/large_file.csv")
        
        stats = get_stats()
        assert stats['memory_change_mb'] < 100, f"Memory usage too high: {stats['memory_change_mb']} MB"

def test_compressed_cache_efficiency():
    """Test that compressed cache provides good compression."""
    from src.utils.compressed_cache import CompressedCache
    
    cache = CompressedCache()
    test_data = np.random.rand(10000, 100)  # Large array
    
    cache.put("test_array", test_data)
    stats = cache.get_stats()
    
    assert stats['compression_ratio'] < 0.8, f"Poor compression ratio: {stats['compression_ratio']}"
    
    retrieved_data = cache.get("test_array")
    assert np.array_equal(test_data, retrieved_data), "Data corruption during compression"

def test_memory_leak_detection():
    """Test that memory leak detection works."""
    from src.utils.memory_profiler import get_memory_profiler
    
    profiler = get_memory_profiler()
    profiler.start_profiling("leak_test")
    
    # Simulate memory leak
    leak_data = []
    for i in range(100):
        leak_data.append(np.random.rand(1000))
    
    result = profiler.stop_profiling()
    
    # Should detect significant memory growth
    assert result.memory_growth_mb > 50, "Failed to detect memory growth"
```

### 3. Performance Monitoring

**Production monitoring:**

```python
def setup_production_monitoring():
    """Setup monitoring for production environment."""
    import logging
    from src.utils.memory_manager import MemoryPressureManager, get_memory_monitor
    
    # Setup logging
    memory_logger = logging.getLogger('memory_monitor')
    handler = logging.FileHandler('logs/memory.log')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    memory_logger.addHandler(handler)
    
    # Setup pressure monitoring
    monitor = get_memory_monitor()
    pressure_manager = MemoryPressureManager(monitor)
    
    def log_memory_pressure():
        stats = monitor.get_current_stats()
        memory_logger.warning(f"Memory pressure detected: {stats.process_mb:.1f} MB")
    
    pressure_manager.register_pressure_callback('medium', log_memory_pressure)
    pressure_manager.register_pressure_callback('high', log_memory_pressure)
    
    # Start monitoring
    monitor.start_monitoring()
    
    return monitor, pressure_manager

# Call during application startup
if __name__ == "__main__":
    monitor, pressure_manager = setup_production_monitoring()
    # Start your application
    run_application()
```

## 🎯 Implementation Checklist

### Phase 1 (Day 1):
- [ ] Update FileIOService initialization with streaming
- [ ] Add memory monitoring to application startup
- [ ] Enable compressed caching in CacheService
- [ ] Update file loading calls to use samples/streaming
- [ ] Add basic memory optimization to data processing

### Phase 2 (Week 1):
- [ ] Add memory profiling to debug performance issues
- [ ] Implement chunked processing for large datasets
- [ ] Setup memory pressure handling
- [ ] Add memory optimization to all data operations
- [ ] Test with production-sized datasets

### Phase 3 (Week 2):
- [ ] Add memory status widget to UI
- [ ] Create cache statistics display
- [ ] Implement smart file loading dialog
- [ ] Add memory warnings to user interface
- [ ] Test UI responsiveness under memory pressure

### Phase 4 (Ongoing):
- [ ] Create configuration management system
- [ ] Add automated memory performance tests
- [ ] Setup production monitoring and logging
- [ ] Monitor and tune performance in production
- [ ] Regularly review and optimize memory patterns

## 🚨 Critical Notes

1. **Backwards Compatibility**: All new methods have fallbacks to existing functionality
2. **Gradual Rollout**: Implement phase by phase to avoid breaking existing features
3. **Testing**: Test with large datasets before deploying to production
4. **Monitoring**: Always monitor memory usage during initial deployment
5. **Configuration**: Make memory limits configurable based on target system specs

The memory optimization system is designed to be implemented incrementally, ensuring your application continues to work while gaining significant performance improvements.