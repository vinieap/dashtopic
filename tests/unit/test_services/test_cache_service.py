"""
Unit tests for CacheService.

This module tests the cache service functionality including:
- Cache key generation
- Embedding storage and retrieval
- Cache size management
- Cache validation
- Cleanup operations

Run with: pytest tests/unit/test_services/test_cache_service.py -v
"""
import pytest
import numpy as np
from pathlib import Path

from src.services.cache_service import CacheService
from src.models.data_models import DataConfig, FileMetadata, CacheInfo


class TestCacheService:
    """Test suite for CacheService class."""
    
    def test_init_creates_cache_directory(self, temp_dir):
        """Test that initialization creates cache directory."""
        cache_dir = temp_dir / "cache"
        service = CacheService(cache_dir=str(cache_dir))
        
        assert cache_dir.exists()
        assert service.cache_dir == cache_dir
        assert service.cache_index_file == cache_dir / "cache_index.json"
    
    def test_init_with_default_cache_dir(self):
        """Test initialization with default cache directory."""
        service = CacheService()
        
        expected_dir = Path.home() / ".bertopic_app" / "cache"
        assert service.cache_dir == expected_dir
    
    def test_cache_key_generation_is_deterministic(self, mock_data_config, mock_model_info, temp_dir):
        """Test that same inputs always generate same cache key."""
        service = CacheService(cache_dir=str(temp_dir))
        
        # Generate key twice with same inputs
        key1 = service.generate_cache_key(mock_data_config, mock_model_info)
        key2 = service.generate_cache_key(mock_data_config, mock_model_info)
        
        assert key1 == key2, "Same inputs should generate same cache key"
        assert len(key1) == 16, "Cache key should be 16 characters"
        assert key1.replace('_', '').replace('-', '').isalnum(), "Cache key should be alphanumeric"
    
    def test_cache_key_different_for_different_inputs(self, mock_model_info, temp_dir):
        """Test that different inputs generate different cache keys."""
        service = CacheService(cache_dir=str(temp_dir))
        
        # Create two different data configs
        file_metadata1 = FileMetadata(
            file_path="/path/to/file1.csv",
            file_name="file1.csv",
            file_size_bytes=1000,
            row_count=100,
            column_count=3
        )
        
        file_metadata2 = FileMetadata(
            file_path="/path/to/file2.csv",
            file_name="file2.csv",
            file_size_bytes=2000,
            row_count=200,
            column_count=3
        )
        
        config1 = DataConfig(
            file_metadata=file_metadata1,
            selected_columns=['text']
        )
        
        config2 = DataConfig(
            file_metadata=file_metadata2,
            selected_columns=['text']
        )
        
        key1 = service.generate_cache_key(config1, mock_model_info)
        key2 = service.generate_cache_key(config2, mock_model_info)
        
        assert key1 != key2, "Different inputs should generate different cache keys"
    
    def test_get_cached_embeddings_returns_none_for_missing_key(self, temp_dir):
        """Test that getting non-existent cache returns None."""
        service = CacheService(cache_dir=str(temp_dir))
        
        result = service.get_cached_embeddings("non_existent_key")
        assert result is None
    
    def test_save_and_retrieve_embeddings(self, temp_dir, sample_embeddings, sample_texts):
        """Test saving and retrieving embeddings from cache."""
        service = CacheService(cache_dir=str(temp_dir))
        cache_key = "test_key_123"
        
        # Save embeddings
        success = service.save_embeddings(cache_key, sample_embeddings, sample_texts)
        assert success is True
        
        # Retrieve embeddings
        result = service.get_cached_embeddings(cache_key)
        assert result is not None
        
        retrieved_embeddings, retrieved_texts = result
        
        # Check embeddings match
        np.testing.assert_array_equal(retrieved_embeddings, sample_embeddings)
        assert retrieved_texts == sample_texts
    
    def test_cache_info_stored_correctly(self, temp_dir, sample_embeddings, sample_texts):
        """Test that cache info is stored correctly."""
        service = CacheService(cache_dir=str(temp_dir))
        cache_key = "test_key_456"
        
        # Save embeddings
        service.save_embeddings(cache_key, sample_embeddings, sample_texts)
        
        # Check cache info
        cache_info = service.get_cache_info(cache_key)
        assert cache_info is not None
        assert cache_info.cache_key == cache_key
        assert cache_info.embedding_shape == sample_embeddings.shape
        assert cache_info.creation_time is not None
        assert cache_info.file_size_bytes > 0
    
    def test_cache_size_calculation(self, temp_dir, sample_embeddings, sample_texts):
        """Test cache size calculation."""
        service = CacheService(cache_dir=str(temp_dir))
        
        # Initial size should be 0
        initial_size = service.get_cache_size()
        assert initial_size >= 0
        
        # Save some embeddings
        service.save_embeddings("key1", sample_embeddings, sample_texts)
        
        # Size should increase
        new_size = service.get_cache_size()
        assert new_size > initial_size
    
    def test_cache_cleanup_by_size(self, temp_dir):
        """Test cache cleanup when size limit is exceeded."""
        # Set very small cache size limit
        service = CacheService(cache_dir=str(temp_dir), max_cache_size_gb=0.001)  # 1 MB
        
        # Create large embeddings that will exceed limit
        large_embeddings = np.random.rand(1000, 384).astype(np.float32)
        texts = [f"Text {i}" for i in range(1000)]
        
        # Save multiple cache entries
        service.save_embeddings("key1", large_embeddings, texts)
        service.save_embeddings("key2", large_embeddings, texts)
        
        # Check that cleanup occurred
        cache_size = service.get_cache_size()
        max_size_bytes = int(0.001 * 1024 * 1024 * 1024)
        
        # Cache should be within limits (allowing some tolerance)
        assert cache_size <= max_size_bytes * 1.1  # 10% tolerance
    
    def test_clear_cache(self, temp_dir, sample_embeddings, sample_texts):
        """Test clearing entire cache."""
        service = CacheService(cache_dir=str(temp_dir))
        
        # Save some embeddings
        service.save_embeddings("key1", sample_embeddings, sample_texts)
        service.save_embeddings("key2", sample_embeddings, sample_texts)
        
        # Verify cache has content
        assert service.get_cache_size() > 0
        assert len(service._cache_index) > 0
        
        # Clear cache
        service.clear_cache()
        
        # Verify cache is empty
        assert service.get_cache_size() == 0
        assert len(service._cache_index) == 0
    
    def test_cache_validation_detects_corrupted_files(self, temp_dir):
        """Test that cache validation detects corrupted files."""
        service = CacheService(cache_dir=str(temp_dir))
        
        # Create a cache entry manually with corrupted file
        cache_key = "corrupted_key"
        cache_file = service.cache_dir / f"{cache_key}.pkl"
        
        # Write invalid data
        cache_file.write_text("This is not valid pickle data")
        
        # Add to index
        from datetime import datetime
        cache_info = CacheInfo(
            cache_key=cache_key,
            file_path=str(cache_file),
            creation_time=datetime.now(),
            last_accessed=datetime.now(),
            file_size_bytes=100,
            embedding_shape=(10, 384),
            model_name="test-model",
            data_hash="test-hash"
        )
        service._cache_index[cache_key] = cache_info
        
        # Validation should remove corrupted entry
        service._validate_cache_files()
        
        assert cache_key not in service._cache_index
        assert not cache_file.exists()
    
    @pytest.mark.parametrize("file_extension", [
        ".csv",
        ".xlsx", 
        ".parquet",
    ])
    def test_cache_key_includes_file_type(self, temp_dir, mock_model_info, file_extension):
        """Test that cache key reflects file type."""
        service = CacheService(cache_dir=str(temp_dir))
        
        file_metadata = FileMetadata(
            file_path=f"/path/to/test{file_extension}",
            file_name=f"test{file_extension}",
            file_size_bytes=1000,
            row_count=100,
            column_count=3
        )
        
        data_config = DataConfig(
            file_metadata=file_metadata,
            selected_columns=['text']
        )
        
        cache_key = service.generate_cache_key(data_config, mock_model_info)
        
        # Key should be different for different file types
        assert len(cache_key) == 16
        assert cache_key.replace('_', '').replace('-', '').isalnum()
    
    def test_cache_key_includes_preprocessing_steps(self, temp_dir, mock_file_metadata, mock_model_info):
        """Test that preprocessing steps affect cache key."""
        service = CacheService(cache_dir=str(temp_dir))
        
        config1 = DataConfig(
            file_metadata=mock_file_metadata,
            selected_columns=['text'],
            preprocessing_steps=[]
        )
        
        config2 = DataConfig(
            file_metadata=mock_file_metadata,
            selected_columns=['text'],
            preprocessing_steps=['lowercase', 'remove_punctuation']
        )
        
        key1 = service.generate_cache_key(config1, mock_model_info)
        key2 = service.generate_cache_key(config2, mock_model_info)
        
        assert key1 != key2, "Different preprocessing should generate different keys"
    
    def test_error_handling_for_invalid_embeddings(self, temp_dir):
        """Test error handling when saving invalid embeddings."""
        service = CacheService(cache_dir=str(temp_dir))
        
        # Try to save invalid embeddings (not numpy array)
        invalid_embeddings = [[1, 2, 3], [4, 5, 6]]  # List instead of numpy array
        texts = ["text1", "text2"]
        
        # Should handle gracefully
        success = service.save_embeddings("test_key", invalid_embeddings, texts)
        assert success is False
    
    def test_error_handling_for_mismatched_dimensions(self, temp_dir):
        """Test error handling for mismatched embeddings and texts."""
        service = CacheService(cache_dir=str(temp_dir))
        
        # Create mismatched data
        embeddings = np.random.rand(5, 384)  # 5 embeddings
        texts = ["text1", "text2", "text3"]  # 3 texts
        
        # Should handle gracefully
        success = service.save_embeddings("test_key", embeddings, texts)
        assert success is False


class TestCacheServiceIntegration:
    """Integration tests for CacheService with real file operations."""
    
    def test_full_cache_workflow(self, temp_dir, mock_data_config, mock_model_info):
        """Test complete cache workflow from key generation to retrieval."""
        service = CacheService(cache_dir=str(temp_dir))
        
        # Generate cache key
        cache_key = service.generate_cache_key(mock_data_config, mock_model_info)
        
        # Check not cached initially
        assert service.get_cached_embeddings(cache_key) is None
        
        # Create and save embeddings
        embeddings = np.random.rand(50, 384).astype(np.float32)
        texts = [f"Document {i}" for i in range(50)]
        
        success = service.save_embeddings(cache_key, embeddings, texts)
        assert success is True
        
        # Retrieve and verify
        result = service.get_cached_embeddings(cache_key)
        assert result is not None
        
        retrieved_embeddings, retrieved_texts = result
        np.testing.assert_array_equal(retrieved_embeddings, embeddings)
        assert retrieved_texts == texts
        
        # Check cache info
        cache_info = service.get_cache_info(cache_key)
        assert cache_info.embedding_shape == (50, 384)
    
    def test_cache_persistence_across_service_instances(self, temp_dir, sample_embeddings, sample_texts):
        """Test that cache persists across service instances."""
        cache_dir = str(temp_dir)
        cache_key = "persistent_key"
        
        # Save with first service instance
        service1 = CacheService(cache_dir=cache_dir)
        service1.save_embeddings(cache_key, sample_embeddings, sample_texts)
        
        # Create new service instance
        service2 = CacheService(cache_dir=cache_dir)
        
        # Should be able to retrieve from new instance
        result = service2.get_cached_embeddings(cache_key)
        assert result is not None
        
        retrieved_embeddings, retrieved_texts = result
        np.testing.assert_array_equal(retrieved_embeddings, sample_embeddings)
        assert retrieved_texts == sample_texts


@pytest.mark.slow
class TestCacheServicePerformance:
    """Performance tests for CacheService."""
    
    def test_cache_performance_large_embeddings(self, temp_dir, benchmark):
        """Benchmark cache operations with large embeddings."""
        service = CacheService(cache_dir=str(temp_dir))
        
        # Create large embeddings
        embeddings = np.random.rand(10000, 384).astype(np.float32)
        texts = [f"Document {i}" for i in range(10000)]
        cache_key = "large_embeddings"
        
        # Benchmark save operation
        def save_large_embeddings():
            return service.save_embeddings(cache_key, embeddings, texts)
        
        result = benchmark(save_large_embeddings)
        assert result is True
        
        # Benchmark retrieval
        def retrieve_large_embeddings():
            return service.get_cached_embeddings(cache_key)
        
        retrieved = benchmark(retrieve_large_embeddings)
        assert retrieved is not None
    
    def test_cache_key_generation_performance(self, benchmark, mock_data_config, mock_model_info, temp_dir):
        """Benchmark cache key generation."""
        service = CacheService(cache_dir=str(temp_dir))
        
        def generate_key():
            return service.generate_cache_key(mock_data_config, mock_model_info)
        
        cache_key = benchmark(generate_key)
        assert len(cache_key) == 16