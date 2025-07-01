"""
Simple working test examples that match the actual API.

These tests demonstrate the testing setup with actual working examples
that you can build upon as you learn the testing system.
"""
import pytest
import pandas as pd
import numpy as np


class TestBasicWorkingExamples:
    """Simple tests that actually work with the current codebase."""
    
    def test_basic_imports_work(self):
        """Test that we can import the main modules."""
        from src.services.file_io_service import FileIOService
        from src.services.cache_service import CacheService
        from src.models.data_models import FileMetadata, DataConfig, ModelInfo
        
        # If we get here, imports work
        assert True
    
    def test_file_io_service_initialization(self):
        """Test that FileIOService can be created."""
        from src.services.file_io_service import FileIOService
        
        service = FileIOService()
        assert service is not None
        assert hasattr(service, 'load_file')
        assert hasattr(service, 'detect_file_format')
    
    def test_cache_service_initialization(self, temp_dir):
        """Test that CacheService can be created."""
        from src.services.cache_service import CacheService
        
        service = CacheService(cache_dir=str(temp_dir))
        assert service is not None
        assert service.cache_dir == temp_dir
    
    def test_file_format_detection(self):
        """Test file format detection with actual API."""
        from src.services.file_io_service import FileIOService
        
        service = FileIOService()
        
        # Test actual method signatures
        assert service.detect_file_format("test.csv") == "csv"
        assert service.detect_file_format("test.xlsx") == "excel"
        assert service.detect_file_format("test.parquet") == "parquet"
    
    def test_cache_key_generation(self, temp_dir):
        """Test cache key generation with correct parameters."""
        from src.services.cache_service import CacheService
        from src.models.data_models import DataConfig, ModelInfo, FileMetadata
        
        service = CacheService(cache_dir=str(temp_dir))
        
        # Create proper FileMetadata with all required fields
        file_metadata = FileMetadata(
            file_path="/test/file.csv",
            file_name="file.csv", 
            file_size_bytes=1000,
            file_format="csv",
            row_count=100,
            column_count=3,
            columns=["text", "category", "id"],
            data_types={"text": "object", "category": "object", "id": "int64"}
        )
        
        # Create DataConfig
        data_config = DataConfig(
            file_metadata=file_metadata,
            selected_columns=["text"]
        )
        
        # Create ModelInfo
        model_info = ModelInfo(
            model_name="test-model",
            model_path="/path/to/model",
            model_type="sentence-transformers"
        )
        
        # Generate cache key
        key1 = service.generate_cache_key(data_config, model_info)
        key2 = service.generate_cache_key(data_config, model_info)
        
        # Same inputs should generate same key
        assert key1 == key2
        assert len(key1) == 16
    
    def test_load_sample_csv(self, test_data_dir):
        """Test loading the sample CSV file we created."""
        from src.services.file_io_service import FileIOService
        
        service = FileIOService()
        sample_file = test_data_dir / "sample_data.csv"
        
        # Load the file
        df, metadata = service.load_file(str(sample_file))
        
        # Basic assertions
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10  # Our sample has 10 rows
        assert "text" in df.columns
        assert "category" in df.columns
        
        # Metadata assertions
        assert metadata.file_name == "sample_data.csv"
        assert metadata.row_count == 10
        assert metadata.file_format == "csv"
    
    def test_embedding_cache_workflow(self, temp_dir):
        """Test a simplified embedding cache workflow."""
        from src.services.cache_service import CacheService
        from src.models.data_models import DataConfig, ModelInfo, FileMetadata
        
        service = CacheService(cache_dir=str(temp_dir))
        
        # Create test data
        file_metadata = FileMetadata(
            file_path="/test/file.csv",
            file_name="file.csv",
            file_size_bytes=1000,
            file_format="csv", 
            row_count=3,
            column_count=1,
            columns=["text"],
            data_types={"text": "object"}
        )
        
        data_config = DataConfig(
            file_metadata=file_metadata,
            selected_columns=["text"]
        )
        
        model_info = ModelInfo(
            model_name="test-model",
            model_path="/path/to/model",
            model_type="sentence-transformers"
        )
        
        # Generate cache key
        cache_key = service.generate_cache_key(data_config, model_info)
        
        # Check cache miss initially
        cached_data = service.get_cached_embeddings(cache_key)
        assert cached_data is None
        
        # Create test embeddings
        embeddings = np.random.rand(3, 384).astype(np.float32)
        texts = ["Text 1", "Text 2", "Text 3"]
        data_hash = "test_hash_123"
        
        # Save to cache (using correct API)
        success = service.save_embeddings(cache_key, embeddings, texts, model_info, data_hash)
        assert success is True
        
        # Retrieve from cache
        cached_data = service.get_cached_embeddings(cache_key)
        assert cached_data is not None
        
        cached_embeddings, cached_texts = cached_data
        assert cached_embeddings.shape == embeddings.shape
        assert cached_texts == texts
    
    def test_data_models_creation(self):
        """Test creating data model instances."""
        from src.models.data_models import FileMetadata, DataConfig, ModelInfo
        
        # Test FileMetadata creation
        metadata = FileMetadata(
            file_path="/test.csv",
            file_name="test.csv",
            file_size_bytes=1024,
            file_format="csv",
            row_count=100,
            column_count=2,
            columns=["col1", "col2"],
            data_types={"col1": "object", "col2": "int64"}
        )
        
        assert metadata.file_size_mb == 1024 / (1024 * 1024)
        assert metadata.text_columns == ["col1"]  # Only object columns
        
        # Test DataConfig creation
        config = DataConfig(
            file_metadata=metadata,
            selected_columns=["col1"]
        )
        
        assert config.is_configured is True
        
        # Test ModelInfo creation
        model = ModelInfo(
            model_name="test-model",
            model_path="/path/to/model",
            model_type="sentence-transformers",
            embedding_dimension=384
        )
        
        assert model.display_name == "Test Model"  # Formatted name
    
    def test_error_handling_examples(self):
        """Test basic error handling."""
        from src.services.file_io_service import FileIOService
        
        service = FileIOService()
        
        # Test unsupported file format
        with pytest.raises(ValueError, match="Unsupported file format"):
            service.detect_file_format("test.txt")
        
        # Test non-existent file
        with pytest.raises(ValueError, match="File not found"):
            service.load_file("non_existent_file.csv")


class TestAdvancedWorkingExamples:
    """More advanced tests showing real usage patterns."""
    
    def test_complete_file_to_cache_workflow(self, temp_dir, test_data_dir):
        """Test complete workflow from file loading to caching."""
        from src.services.file_io_service import FileIOService
        from src.services.cache_service import CacheService
        from src.models.data_models import DataConfig, ModelInfo
        
        # Initialize services
        file_service = FileIOService()
        cache_service = CacheService(cache_dir=str(temp_dir))
        
        # Load sample data
        sample_file = test_data_dir / "sample_data.csv"
        df, metadata = file_service.load_file(str(sample_file))
        
        # Create configuration
        data_config = DataConfig(
            file_metadata=metadata,
            selected_columns=["text"]
        )
        
        model_info = ModelInfo(
            model_name="all-MiniLM-L6-v2",
            model_path="/models/all-MiniLM-L6-v2",
            model_type="sentence-transformers",
            embedding_dimension=384
        )
        
        # Generate cache key
        cache_key = cache_service.generate_cache_key(data_config, model_info)
        
        # Simulate embedding generation
        num_docs = len(df)
        embeddings = np.random.rand(num_docs, 384).astype(np.float32)
        texts = df["text"].tolist()
        data_hash = "sample_data_hash"
        
        # Save to cache
        success = cache_service.save_embeddings(cache_key, embeddings, texts, model_info, data_hash)
        assert success is True
        
        # Verify retrieval
        cached_data = cache_service.get_cached_embeddings(cache_key)
        assert cached_data is not None
        
        cached_embeddings, cached_texts = cached_data
        assert cached_embeddings.shape == (num_docs, 384)
        assert len(cached_texts) == num_docs
    
    @pytest.mark.parametrize("file_format,extension", [
        ("csv", ".csv"),
        ("excel", ".xlsx"),
        ("parquet", ".parquet"), 
        ("feather", ".feather")
    ])
    def test_multiple_file_formats(self, file_format, extension):
        """Test file format detection for multiple formats."""
        from src.services.file_io_service import FileIOService
        
        service = FileIOService()
        
        result = service.detect_file_format(f"test{extension}")
        assert result == file_format
    
    def test_cache_different_configurations(self, temp_dir):
        """Test that different configurations generate different cache keys."""
        from src.services.cache_service import CacheService
        from src.models.data_models import DataConfig, ModelInfo, FileMetadata
        
        service = CacheService(cache_dir=str(temp_dir))
        
        base_metadata = FileMetadata(
            file_path="/test.csv",
            file_name="test.csv",
            file_size_bytes=1000,
            file_format="csv",
            row_count=100, 
            column_count=2,
            columns=["text", "category"],
            data_types={"text": "object", "category": "object"}
        )
        
        model_info = ModelInfo(
            model_name="test-model",
            model_path="/path/to/model",
            model_type="sentence-transformers"
        )
        
        # Different column selections should generate different keys
        config1 = DataConfig(file_metadata=base_metadata, selected_columns=["text"])
        config2 = DataConfig(file_metadata=base_metadata, selected_columns=["category"])
        config3 = DataConfig(file_metadata=base_metadata, selected_columns=["text", "category"])
        
        key1 = service.generate_cache_key(config1, model_info)
        key2 = service.generate_cache_key(config2, model_info)
        key3 = service.generate_cache_key(config3, model_info)
        
        # All keys should be different
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3
        
        # But same config should generate same key
        key1_duplicate = service.generate_cache_key(config1, model_info)
        assert key1 == key1_duplicate