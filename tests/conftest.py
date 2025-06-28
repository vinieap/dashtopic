"""
Pytest configuration for BERTopic Desktop Application tests.
"""
import pytest
import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

@pytest.fixture
def app_config():
    """Fixture providing test application configuration."""
    return {
        "test_mode": True,
        "cache_dir": "test_cache",
        "log_level": "DEBUG"
    }

@pytest.fixture
def sample_data():
    """Fixture providing sample test data."""
    return [
        "This is a sample document about machine learning.",
        "Natural language processing is a fascinating field.",
        "Topic modeling helps discover hidden themes in documents.",
        "BERTopic is a powerful tool for topic analysis."
    ] 