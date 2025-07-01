"""
Integration tests for data workflow.

This module tests the integration between:
- File I/O Service
- Data Validation Service  
- Cache Service
- Data processing pipeline

Run with: pytest tests/integration/test_data_workflow.py -v
"""
import pytest
import pandas as pd
from pathlib import Path

from src.services.file_io_service import FileIOService
from src.services.data_validation_service import DataValidationService
from src.services.cache_service import CacheService
from src.models.data_models import DataConfig, FileMetadata


@pytest.mark.integration
class TestDataWorkflowIntegration:
    """Integration tests for the complete data workflow."""
    
    def test_complete_data_loading_workflow(self, temp_dir, test_data_dir):
        """Test complete workflow from file loading to validation."""
        # Initialize services
        file_service = FileIOService()
        validation_service = DataValidationService()
        
        # Use the sample data file
        sample_file = test_data_dir / "sample_data.csv"
        
        # Step 1: Load file
        df, metadata = file_service.load_file(str(sample_file))
        
        # Verify file loaded correctly
        assert len(df) == 10
        assert 'text' in df.columns
        assert 'category' in df.columns
        
        # Step 2: Create data configuration
        data_config = DataConfig(
            file_metadata=metadata,
            selected_columns=['text'],
            text_combination_method='concatenate',
            text_combination_separator=' ',
            include_column_names=False,
            remove_empty_rows=True,
            min_text_length=10,
            max_text_length=1000
        )
        
        # Step 3: Validate data configuration
        validation_result = validation_service.validate_data_config(df, data_config)
        
        # Verify validation passed
        assert validation_result.is_valid
        assert len(validation_result.errors) == 0
        
        # Step 4: Process text data
        texts = validation_service.extract_texts(df, data_config)
        
        # Verify text extraction
        assert len(texts) == 10
        assert all(isinstance(text, str) for text in texts)
        assert all(len(text) >= data_config.min_text_length for text in texts)
    
    def test_cache_integration_workflow(self, temp_dir, mock_model_info):
        """Test workflow with caching integration."""
        # Initialize services
        file_service = FileIOService()
        cache_service = CacheService(cache_dir=str(temp_dir / "cache"))
        
        # Create test data file
        test_df = pd.DataFrame({
            'text': [
                'Test document about machine learning',
                'Another document about data science',
                'Third document about artificial intelligence'
            ],
            'id': [1, 2, 3]
        })
        
        test_file = temp_dir / "test.csv"
        test_df.to_csv(test_file, index=False)
        
        # Load data
        df, metadata = file_service.load_file(str(test_file))
        
        # Create configuration
        data_config = DataConfig(
            file_metadata=metadata,
            selected_columns=['text']
        )
        
        # Generate cache key
        cache_key = cache_service.generate_cache_key(data_config, mock_model_info)
        
        # Verify cache key is generated
        assert cache_key is not None
        assert len(cache_key) == 16
        
        # Verify no cached data initially
        cached_data = cache_service.get_cached_embeddings(cache_key)
        assert cached_data is None
        
        # Simulate saving embeddings
        import numpy as np
        dummy_embeddings = np.random.rand(3, 384)
        texts = df['text'].tolist()
        
        success = cache_service.save_embeddings(cache_key, dummy_embeddings, texts)
        assert success is True
        
        # Verify cached data can be retrieved
        cached_data = cache_service.get_cached_embeddings(cache_key)
        assert cached_data is not None
        
        cached_embeddings, cached_texts = cached_data
        assert cached_embeddings.shape == dummy_embeddings.shape
        assert cached_texts == texts
    
    def test_data_validation_with_different_configurations(self, sample_csv_file):
        """Test data validation with various configurations."""
        file_service = FileIOService()
        validation_service = DataValidationService()
        
        # Load sample data
        df, metadata = file_service.load_file(str(sample_csv_file))
        
        # Test different column selections
        configs_to_test = [
            {
                'selected_columns': ['text'],
                'description': 'Single text column'
            },
            {
                'selected_columns': ['text', 'category'],
                'description': 'Multiple columns'
            },
            {
                'selected_columns': ['category'],
                'description': 'Non-text column only'
            }
        ]
        
        for config_data in configs_to_test:
            data_config = DataConfig(
                file_metadata=metadata,
                selected_columns=config_data['selected_columns']
            )
            
            validation_result = validation_service.validate_data_config(df, data_config)
            
            # All configurations should be valid for our test data
            assert validation_result is not None
            
            # Extract texts if validation passed
            if validation_result.is_valid:
                texts = validation_service.extract_texts(df, data_config)
                assert len(texts) > 0
    
    def test_error_handling_in_workflow(self, temp_dir):
        """Test error handling throughout the workflow."""
        file_service = FileIOService()
        validation_service = DataValidationService()
        
        # Test with invalid file
        with pytest.raises(FileNotFoundError):
            file_service.load_file("non_existent_file.csv")
        
        # Test with empty file
        empty_file = temp_dir / "empty.csv"
        empty_file.write_text("")
        
        with pytest.raises(ValueError):
            file_service.load_file(str(empty_file))
        
        # Test validation with invalid configuration
        valid_df = pd.DataFrame({'text': ['sample text']})
        invalid_metadata = FileMetadata(
            file_path="test.csv",
            file_name="test.csv",
            file_size_bytes=100,
            row_count=1,
            column_count=1
        )
        
        invalid_config = DataConfig(
            file_metadata=invalid_metadata,
            selected_columns=['non_existent_column']  # Column doesn't exist
        )
        
        validation_result = validation_service.validate_data_config(valid_df, invalid_config)
        assert not validation_result.is_valid
        assert len(validation_result.errors) > 0
    
    def test_performance_with_larger_dataset(self, temp_dir):
        """Test workflow performance with larger dataset."""
        import pandas as pd
        import time
        
        # Create larger test dataset
        large_df = pd.DataFrame({
            'text': [f'Document {i} with some content about topic {i % 10}' for i in range(1000)],
            'category': [f'Category_{i % 5}' for i in range(1000)],
            'id': range(1000)
        })
        
        large_file = temp_dir / "large_test.csv"
        large_df.to_csv(large_file, index=False)
        
        # Initialize services
        file_service = FileIOService()
        validation_service = DataValidationService()
        
        # Measure workflow performance
        start_time = time.time()
        
        # Load data
        df, metadata = file_service.load_file(str(large_file))
        
        # Create configuration
        data_config = DataConfig(
            file_metadata=metadata,
            selected_columns=['text']
        )
        
        # Validate
        validation_result = validation_service.validate_data_config(df, data_config)
        
        # Extract texts
        if validation_result.is_valid:
            texts = validation_service.extract_texts(df, data_config)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Verify results
        assert len(df) == 1000
        assert validation_result.is_valid
        assert len(texts) == 1000
        
        # Performance assertion (should complete within reasonable time)
        assert duration < 5.0, f"Workflow took too long: {duration:.2f} seconds"


@pytest.mark.integration
@pytest.mark.slow
class TestDataWorkflowPerformance:
    """Performance tests for integrated data workflow."""
    
    def test_workflow_memory_usage(self, temp_dir, benchmark):
        """Test memory usage of complete workflow."""
        import pandas as pd
        import psutil
        import os
        
        # Create test data
        test_df = pd.DataFrame({
            'text': [f'Test document {i} with content' for i in range(5000)],
            'metadata': [f'Meta {i}' for i in range(5000)]
        })
        
        test_file = temp_dir / "memory_test.csv"
        test_df.to_csv(test_file, index=False)
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        def run_workflow():
            file_service = FileIOService()
            validation_service = DataValidationService()
            
            df, metadata = file_service.load_file(str(test_file))
            
            data_config = DataConfig(
                file_metadata=metadata,
                selected_columns=['text']
            )
            
            validation_result = validation_service.validate_data_config(df, data_config)
            texts = validation_service.extract_texts(df, data_config)
            
            return len(texts)
        
        # Benchmark the workflow
        result = benchmark(run_workflow)
        
        # Check memory didn't grow excessively
        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        assert result == 5000
        assert memory_growth < 100, f"Memory grew by {memory_growth:.1f} MB"
    
    def test_concurrent_workflow_execution(self, temp_dir):
        """Test multiple workflows running concurrently."""
        import threading
        import pandas as pd
        
        # Create multiple test files
        results = {}
        
        def run_single_workflow(file_id):
            """Run workflow for a single file."""
            try:
                # Create test data
                test_df = pd.DataFrame({
                    'text': [f'File {file_id} document {i}' for i in range(100)],
                    'id': range(100)
                })
                
                test_file = temp_dir / f"concurrent_test_{file_id}.csv"
                test_df.to_csv(test_file, index=False)
                
                # Run workflow
                file_service = FileIOService()
                validation_service = DataValidationService()
                
                df, metadata = file_service.load_file(str(test_file))
                
                data_config = DataConfig(
                    file_metadata=metadata,
                    selected_columns=['text']
                )
                
                validation_result = validation_service.validate_data_config(df, data_config)
                texts = validation_service.extract_texts(df, data_config)
                
                results[file_id] = {
                    'success': True,
                    'text_count': len(texts),
                    'valid': validation_result.is_valid
                }
                
            except Exception as e:
                results[file_id] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Run 5 workflows concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=run_single_workflow, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all workflows completed successfully
        assert len(results) == 5
        for file_id, result in results.items():
            assert result['success'], f"Workflow {file_id} failed: {result.get('error')}"
            assert result['text_count'] == 100
            assert result['valid'] is True