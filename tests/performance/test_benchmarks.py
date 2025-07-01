"""
Performance benchmarks for BERTopic Desktop Application.

This module contains benchmark tests for:
- File loading performance
- Data processing speed
- Cache operations
- Memory usage
- UI responsiveness

Run with: pytest tests/performance/test_benchmarks.py -v
Or with specific benchmarks: pytest tests/performance/test_benchmarks.py::test_file_loading_benchmark -v
"""
import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
from pathlib import Path

from src.services.file_io_service import FileIOService
from src.services.cache_service import CacheService
from src.models.data_models import DataConfig, FileMetadata


@pytest.mark.benchmark
class TestFileIOBenchmarks:
    """Benchmark tests for file I/O operations."""
    
    def test_csv_loading_benchmark(self, benchmark, temp_dir):
        """Benchmark CSV file loading performance."""
        # Create test CSV with various sizes
        sizes = [1000, 5000, 10000]
        
        for size in sizes:
            # Create test data
            test_df = pd.DataFrame({
                'id': range(size),
                'text': [f'Document {i} with some longer text content for realistic testing' for i in range(size)],
                'category': [f'Category_{i % 10}' for i in range(size)],
                'score': np.random.rand(size),
                'timestamp': pd.date_range('2024-01-01', periods=size, freq='H')
            })
            
            csv_file = temp_dir / f"benchmark_{size}.csv"
            test_df.to_csv(csv_file, index=False)
            
            service = FileIOService()
            
            def load_csv():
                return service.load_file(str(csv_file))
            
            result = benchmark(load_csv)
            df, metadata = result
            
            # Verify correct loading
            assert len(df) == size
            assert metadata.row_count == size
            
            print(f"CSV loading benchmark ({size} rows): {benchmark.stats['mean']:.4f}s")
    
    def test_excel_loading_benchmark(self, benchmark, temp_dir):
        """Benchmark Excel file loading performance."""
        # Create test Excel file
        test_df = pd.DataFrame({
            'id': range(5000),
            'text': [f'Excel document {i}' for i in range(5000)],
            'value': np.random.rand(5000)
        })
        
        excel_file = temp_dir / "benchmark.xlsx"
        test_df.to_excel(excel_file, index=False)
        
        service = FileIOService()
        
        def load_excel():
            return service.load_file(str(excel_file))
        
        result = benchmark(load_excel)
        df, metadata = result
        
        assert len(df) == 5000
        print(f"Excel loading benchmark: {benchmark.stats['mean']:.4f}s")
    
    def test_parquet_loading_benchmark(self, benchmark, temp_dir):
        """Benchmark Parquet file loading performance."""
        # Create test Parquet file
        test_df = pd.DataFrame({
            'id': range(10000),
            'text': [f'Parquet document {i}' for i in range(10000)],
            'embedding': [np.random.rand(384).tolist() for _ in range(10000)]
        })
        
        parquet_file = temp_dir / "benchmark.parquet"
        test_df.to_parquet(parquet_file, index=False)
        
        service = FileIOService()
        
        def load_parquet():
            return service.load_file(str(parquet_file))
        
        result = benchmark(load_parquet)
        df, metadata = result
        
        assert len(df) == 10000
        print(f"Parquet loading benchmark: {benchmark.stats['mean']:.4f}s")


@pytest.mark.benchmark
class TestCacheBenchmarks:
    """Benchmark tests for cache operations."""
    
    def test_cache_write_benchmark(self, benchmark, temp_dir):
        """Benchmark cache write performance."""
        service = CacheService(cache_dir=str(temp_dir))
        
        # Create test embeddings
        embeddings = np.random.rand(10000, 384).astype(np.float32)
        texts = [f'Document {i}' for i in range(10000)]
        cache_key = "benchmark_key"
        
        def save_embeddings():
            return service.save_embeddings(cache_key, embeddings, texts)
        
        result = benchmark(save_embeddings)
        assert result is True
        
        print(f"Cache write benchmark (10k embeddings): {benchmark.stats['mean']:.4f}s")
    
    def test_cache_read_benchmark(self, benchmark, temp_dir):
        """Benchmark cache read performance."""
        service = CacheService(cache_dir=str(temp_dir))
        
        # Prepare cached data
        embeddings = np.random.rand(10000, 384).astype(np.float32)
        texts = [f'Document {i}' for i in range(10000)]
        cache_key = "benchmark_read_key"
        
        # Save first
        service.save_embeddings(cache_key, embeddings, texts)
        
        def load_embeddings():
            return service.get_cached_embeddings(cache_key)
        
        result = benchmark(load_embeddings)
        assert result is not None
        
        cached_embeddings, cached_texts = result
        assert cached_embeddings.shape == embeddings.shape
        assert len(cached_texts) == len(texts)
        
        print(f"Cache read benchmark (10k embeddings): {benchmark.stats['mean']:.4f}s")
    
    def test_cache_key_generation_benchmark(self, benchmark, mock_data_config, mock_model_info, temp_dir):
        """Benchmark cache key generation performance."""
        service = CacheService(cache_dir=str(temp_dir))
        
        def generate_key():
            return service.generate_cache_key(mock_data_config, mock_model_info)
        
        result = benchmark(generate_key)
        assert len(result) == 16
        
        print(f"Cache key generation benchmark: {benchmark.stats['mean']:.6f}s")


@pytest.mark.benchmark
class TestDataProcessingBenchmarks:
    """Benchmark tests for data processing operations."""
    
    def test_text_extraction_benchmark(self, benchmark, temp_dir):
        """Benchmark text extraction from DataFrames."""
        # Create large DataFrame
        test_df = pd.DataFrame({
            'text1': [f'First part of document {i}' for i in range(10000)],
            'text2': [f'Second part of document {i}' for i in range(10000)],
            'metadata': [f'Meta {i}' for i in range(10000)]
        })
        
        # Create mock configuration
        file_metadata = FileMetadata(
            file_path="test.csv",
            file_name="test.csv",
            file_size_bytes=1000000,
            row_count=10000,
            column_count=3
        )
        
        data_config = DataConfig(
            file_metadata=file_metadata,
            selected_columns=['text1', 'text2'],
            text_combination_method='concatenate',
            text_combination_separator=' '
        )
        
        def extract_texts():
            # Simulate text extraction
            if data_config.text_combination_method == 'concatenate':
                texts = []
                for _, row in test_df.iterrows():
                    combined_text = data_config.text_combination_separator.join([
                        str(row[col]) for col in data_config.selected_columns
                    ])
                    texts.append(combined_text)
                return texts
            return []
        
        result = benchmark(extract_texts)
        assert len(result) == 10000
        
        print(f"Text extraction benchmark (10k docs): {benchmark.stats['mean']:.4f}s")
    
    def test_dataframe_filtering_benchmark(self, benchmark):
        """Benchmark DataFrame filtering operations."""
        # Create large DataFrame
        test_df = pd.DataFrame({
            'text': [f'Document {i} content' for i in range(50000)],
            'length': [len(f'Document {i} content') for i in range(50000)],
            'valid': np.random.choice([True, False], 50000, p=[0.8, 0.2])
        })
        
        def filter_dataframe():
            # Apply common filters
            filtered = test_df[
                (test_df['length'] >= 10) &
                (test_df['valid'] == True) &
                (test_df['text'].str.len() > 0)
            ]
            return len(filtered)
        
        result = benchmark(filter_dataframe)
        assert result > 0
        
        print(f"DataFrame filtering benchmark (50k rows): {benchmark.stats['mean']:.4f}s")


@pytest.mark.benchmark
class TestMemoryBenchmarks:
    """Benchmark tests for memory usage."""
    
    def test_memory_usage_file_loading(self, temp_dir):
        """Test memory usage during file loading."""
        # Create large test file
        large_df = pd.DataFrame({
            'text': [f'Large document {i} with substantial content' * 10 for i in range(20000)],
            'id': range(20000),
            'metadata': [f'Meta {i}' * 5 for i in range(20000)]
        })
        
        large_file = temp_dir / "memory_test.csv"
        large_df.to_csv(large_file, index=False)
        
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load file
        service = FileIOService()
        df, metadata = service.load_file(str(large_file))
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - initial_memory
        
        # Cleanup
        del df, metadata, large_df
        import gc
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_freed = peak_memory - final_memory
        
        print(f"Memory usage for loading large file: {memory_used:.1f} MB")
        print(f"Memory freed after cleanup: {memory_freed:.1f} MB")
        
        # Assertions
        assert memory_used < 500, f"Memory usage too high: {memory_used:.1f} MB"
        assert memory_freed > memory_used * 0.5, "Insufficient memory cleanup"
    
    def test_memory_usage_cache_operations(self, temp_dir):
        """Test memory usage during cache operations."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        service = CacheService(cache_dir=str(temp_dir))
        
        # Create and cache multiple embedding sets
        for i in range(5):
            embeddings = np.random.rand(5000, 384).astype(np.float32)
            texts = [f'Cache test document {j}' for j in range(5000)]
            cache_key = f"memory_test_key_{i}"
            
            service.save_embeddings(cache_key, embeddings, texts)
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = current_memory - initial_memory
            
            print(f"Memory after caching set {i+1}: {memory_growth:.1f} MB")
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_memory_used = peak_memory - initial_memory
        
        # Clear cache
        service.clear_cache()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_after_clear = final_memory - initial_memory
        
        print(f"Total memory used: {total_memory_used:.1f} MB")
        print(f"Memory after cache clear: {memory_after_clear:.1f} MB")
        
        # Memory should be reasonable
        assert total_memory_used < 1000, f"Cache memory usage too high: {total_memory_used:.1f} MB"


@pytest.mark.benchmark
@pytest.mark.slow
class TestScalabilityBenchmarks:
    """Benchmark tests for scalability with large datasets."""
    
    @pytest.mark.parametrize("size", [1000, 5000, 10000, 25000])
    def test_scalability_by_dataset_size(self, benchmark, temp_dir, size):
        """Test performance scaling with dataset size."""
        # Create test data of specified size
        test_df = pd.DataFrame({
            'text': [f'Scalability test document {i} with content' for i in range(size)],
            'category': [f'Cat_{i % 20}' for i in range(size)],
            'id': range(size)
        })
        
        test_file = temp_dir / f"scalability_{size}.csv"
        test_df.to_csv(test_file, index=False)
        
        service = FileIOService()
        
        def load_and_process():
            df, metadata = service.load_file(str(test_file))
            # Simulate some processing
            text_lengths = df['text'].str.len()
            avg_length = text_lengths.mean()
            return len(df), avg_length
        
        result = benchmark(load_and_process)
        doc_count, avg_length = result
        
        assert doc_count == size
        assert avg_length > 0
        
        # Calculate operations per second
        ops_per_second = size / benchmark.stats['mean']
        
        print(f"Scalability test ({size} docs): {benchmark.stats['mean']:.4f}s ({ops_per_second:.0f} docs/sec)")
    
    def test_concurrent_operations_benchmark(self, temp_dir):
        """Test performance under concurrent operations."""
        import threading
        import time
        
        # Create test files
        num_files = 5
        test_files = []
        
        for i in range(num_files):
            test_df = pd.DataFrame({
                'text': [f'Concurrent test {i} doc {j}' for j in range(1000)],
                'id': range(1000)
            })
            
            test_file = temp_dir / f"concurrent_{i}.csv"
            test_df.to_csv(test_file, index=False)
            test_files.append(test_file)
        
        results = {}
        
        def load_file_worker(file_path, worker_id):
            """Worker function to load a file."""
            start_time = time.time()
            
            service = FileIOService()
            df, metadata = service.load_file(str(file_path))
            
            end_time = time.time()
            duration = end_time - start_time
            
            results[worker_id] = {
                'duration': duration,
                'rows': len(df),
                'success': True
            }
        
        # Launch concurrent workers
        threads = []
        start_time = time.time()
        
        for i, file_path in enumerate(test_files):
            thread = threading.Thread(target=load_file_worker, args=(file_path, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Analyze results
        avg_duration = sum(r['duration'] for r in results.values()) / len(results)
        total_rows = sum(r['rows'] for r in results.values())
        
        print(f"Concurrent operations benchmark:")
        print(f"  Total time: {total_time:.4f}s")
        print(f"  Average per file: {avg_duration:.4f}s")
        print(f"  Total rows processed: {total_rows}")
        print(f"  Throughput: {total_rows/total_time:.0f} rows/sec")
        
        # All operations should succeed
        assert all(r['success'] for r in results.values())
        assert len(results) == num_files