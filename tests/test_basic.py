"""
Basic test to verify pytest setup works correctly.
"""
import pytest


def test_pytest_works():
    """Test that pytest is working correctly."""
    assert True


def test_basic_math():
    """Test basic math operations."""
    assert 2 + 2 == 4
    assert 10 * 5 == 50


def test_string_operations():
    """Test string operations."""
    text = "Hello, World!"
    assert "Hello" in text
    assert len(text) == 13


@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
    (5, 10)
])
def test_parametrized_function(input, expected):
    """Test parametrized testing works."""
    result = input * 2
    assert result == expected


def test_fixtures_work(temp_dir):
    """Test that fixtures work correctly."""
    assert temp_dir.exists()
    assert temp_dir.is_dir()
    
    # Create a test file
    test_file = temp_dir / "test.txt"
    test_file.write_text("Hello from test!")
    
    # Verify file was created
    assert test_file.exists()
    assert test_file.read_text() == "Hello from test!"