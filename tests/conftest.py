"""
Pytest configuration and shared fixtures for all tests.

This file is automatically loaded by pytest and provides:
- Common fixtures used across multiple tests
- Test data generators
- Mock objects and utilities
- Custom assertions and helpers
"""
import sys
import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
import tempfile
import shutil
from faker import Faker
import customtkinter as ctk

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Initialize Faker for test data generation
fake = Faker()

# ============================================================================
# Path Fixtures
# ============================================================================

@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_data_dir(project_root):
    """Return the test data directory."""
    return project_root / "tests" / "fixtures"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir)


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def sample_dataframe():
    """Provide a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': range(1, 101),
        'text': [fake.text(max_nb_chars=200) for _ in range(100)],
        'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'score': np.random.rand(100) * 100,
        'is_valid': np.random.choice([True, False], 100)
    })


@pytest.fixture
def small_dataframe():
    """Provide a small DataFrame for quick tests."""
    return pd.DataFrame({
        'text': ['Document 1 about science', 'Document 2 about technology', 'Document 3 about art'],
        'category': ['Science', 'Tech', 'Art'],
        'length': [24, 27, 20]
    })


@pytest.fixture
def empty_dataframe():
    """Provide an empty DataFrame for edge case testing."""
    return pd.DataFrame()


@pytest.fixture
def sample_embeddings():
    """Provide sample embeddings array."""
    # 100 documents with 384-dimensional embeddings (typical for sentence-transformers)
    return np.random.rand(100, 384).astype(np.float32)


@pytest.fixture
def sample_texts():
    """Provide sample text documents."""
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to interpret visual information.",
        "Reinforcement learning involves learning from interactions."
    ]


# ============================================================================
# File Fixtures
# ============================================================================

@pytest.fixture
def sample_csv_file(temp_dir, sample_dataframe):
    """Create a temporary CSV file with sample data."""
    file_path = temp_dir / "sample_data.csv"
    sample_dataframe.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def sample_excel_file(temp_dir, sample_dataframe):
    """Create a temporary Excel file with sample data."""
    file_path = temp_dir / "sample_data.xlsx"
    sample_dataframe.to_excel(file_path, index=False)
    return file_path


@pytest.fixture
def sample_parquet_file(temp_dir, sample_dataframe):
    """Create a temporary Parquet file with sample data."""
    file_path = temp_dir / "sample_data.parquet"
    sample_dataframe.to_parquet(file_path, index=False)
    return file_path


# ============================================================================
# Model and Configuration Fixtures
# ============================================================================

@pytest.fixture
def mock_file_metadata():
    """Provide mock FileMetadata object."""
    from src.models.data_models import FileMetadata
    
    return FileMetadata(
        file_path="/path/to/test.csv",
        file_name="test.csv",
        file_size_bytes=1024 * 1024,  # 1 MB
        file_format="csv",
        row_count=100,
        column_count=5,
        columns=['text', 'category', 'id', 'score', 'date'],
        data_types={
            'text': 'object',
            'category': 'object', 
            'id': 'int64',
            'score': 'float64',
            'date': 'datetime64[ns]'
        },
        encoding="utf-8",
        delimiter=","
    )


@pytest.fixture
def mock_data_config(mock_file_metadata):
    """Provide mock DataConfig object."""
    from src.models.data_models import DataConfig
    
    return DataConfig(
        file_metadata=mock_file_metadata,
        selected_columns=['text'],
        text_combination_method='concatenate',
        text_combination_separator=' ',
        include_column_names=False,
        remove_empty_rows=True,
        min_text_length=10,
        max_text_length=1000,
        preprocessing_steps=[]
    )


@pytest.fixture
def mock_model_info():
    """Provide mock ModelInfo object."""
    from src.models.data_models import ModelInfo
    
    return ModelInfo(
        model_name="all-MiniLM-L6-v2",
        model_path="/path/to/model",
        model_type="sentence-transformers",
        embedding_dimension=384,
        max_sequence_length=256,
        description="General purpose sentence embeddings",
        model_size_mb=90.0,
        is_loaded=False
    )


# ============================================================================
# Service Mocks
# ============================================================================

@pytest.fixture
def mock_cache_service():
    """Provide a mock CacheService."""
    service = Mock()
    service.generate_cache_key.return_value = "test_cache_key_12345"
    service.get_cached_embeddings.return_value = None
    service.save_embeddings.return_value = True
    service.get_cache_size.return_value = 1024 * 1024  # 1 MB
    return service


@pytest.fixture
def mock_embedding_service():
    """Provide a mock EmbeddingService."""
    service = Mock()
    service.get_available_models.return_value = [
        {"name": "all-MiniLM-L6-v2", "size": 90},
        {"name": "all-mpnet-base-v2", "size": 420}
    ]
    service.load_model.return_value = True
    service.generate_embeddings.return_value = np.random.rand(100, 384)
    return service


@pytest.fixture
def mock_bertopic_service():
    """Provide a mock BERTopicService."""
    service = Mock()
    service.is_training = False
    service.train_model_async = Mock()
    service.get_topics.return_value = [
        {"topic": 0, "words": ["machine", "learning", "ai"], "size": 25},
        {"topic": 1, "words": ["data", "analysis", "science"], "size": 30}
    ]
    return service


# ============================================================================
# UI Testing Fixtures
# ============================================================================

@pytest.fixture
def app_window():
    """Create a test application window."""
    # Set up CustomTkinter for testing
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    
    # Create root window
    root = ctk.CTk()
    root.geometry("800x600")
    root.title("Test Window")
    
    yield root
    
    # Cleanup
    try:
        root.quit()
        root.destroy()
    except:
        pass


# ============================================================================
# Custom Assertions
# ============================================================================

class CustomAssertions:
    """Custom assertion methods for tests."""
    
    @staticmethod
    def assert_dataframe_equal(df1, df2, check_exact=True):
        """Assert two DataFrames are equal."""
        pd.testing.assert_frame_equal(df1, df2, check_exact=check_exact)
    
    @staticmethod
    def assert_embeddings_valid(embeddings, expected_shape=None):
        """Assert embeddings are valid."""
        assert isinstance(embeddings, np.ndarray), "Embeddings should be numpy array"
        assert embeddings.dtype in [np.float32, np.float64], "Embeddings should be float"
        assert len(embeddings.shape) == 2, "Embeddings should be 2D"
        
        if expected_shape:
            assert embeddings.shape == expected_shape, f"Expected shape {expected_shape}, got {embeddings.shape}"


@pytest.fixture
def custom_assertions():
    """Provide custom assertion methods."""
    return CustomAssertions()


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "ui: marks tests that require UI interaction"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks unit tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks performance benchmark tests"
    ) 