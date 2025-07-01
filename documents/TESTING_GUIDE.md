# Complete Testing Guide for BERTopic Desktop Application

## Table of Contents
1. [What is Testing and Why Do We Need It?](#what-is-testing)
2. [Types of Tests](#types-of-tests)
3. [Testing Tools We'll Use](#testing-tools)
4. [Setting Up the Testing Environment](#setup)
5. [Writing Your First Test](#first-test)
6. [Testing Best Practices](#best-practices)
7. [Running Tests](#running-tests)
8. [Understanding Test Coverage](#coverage)
9. [Debugging Failed Tests](#debugging)
10. [Continuous Integration](#ci)

## What is Testing and Why Do We Need It? {#what-is-testing}

Testing is writing code that verifies your application works correctly. Think of it as:
- **Quality Assurance**: Catching bugs before users do
- **Documentation**: Tests show how code should be used
- **Confidence**: Change code without fear of breaking things
- **Time Saver**: Find issues early when they're cheap to fix

### Benefits for Your Project
1. **Reliability**: Know your topic modeling actually works
2. **Refactoring Safety**: Improve code without breaking features
3. **Faster Development**: Catch errors immediately
4. **Better Design**: Writing testable code improves architecture

## Types of Tests {#types-of-tests}

### 1. Unit Tests
Test individual functions/methods in isolation.

**Example**: Testing if the cache key generation works correctly
```python
def test_cache_key_generation():
    # Test that the same inputs always generate the same key
    key1 = generate_cache_key("data.csv", "model1")
    key2 = generate_cache_key("data.csv", "model1")
    assert key1 == key2
```

### 2. Integration Tests
Test how different parts work together.

**Example**: Testing if data loading + validation work together
```python
def test_data_loading_workflow():
    # Load a file
    data = load_file("test.csv")
    # Validate it
    validation_result = validate_data(data)
    assert validation_result.is_valid
```

### 3. UI/End-to-End Tests
Test the complete user workflow through the GUI.

**Example**: Testing the complete topic modeling workflow
```python
def test_complete_workflow(app):
    # Load file through UI
    app.click_button("Browse")
    app.select_file("test.csv")
    # Select columns
    app.select_columns(["text_column"])
    # Run analysis
    app.click_button("Run Topic Modeling")
    # Check results appear
    assert app.has_results()
```

### 4. Performance Tests
Ensure your app performs well.

**Example**: Testing embedding generation speed
```python
def test_embedding_performance():
    start_time = time.time()
    embeddings = generate_embeddings(["text"] * 1000)
    duration = time.time() - start_time
    assert duration < 10  # Should take less than 10 seconds
```

## Testing Tools We'll Use {#testing-tools}

### 1. pytest
The most popular Python testing framework.
- **Why**: Simple, powerful, great error messages
- **Install**: `pip install pytest`

### 2. pytest-cov
Measures how much of your code is tested.
- **Why**: See untested code
- **Install**: `pip install pytest-cov`

### 3. pytest-mock
Makes it easy to mock (fake) parts of your code.
- **Why**: Test in isolation without external dependencies
- **Install**: `pip install pytest-mock`

### 4. pytest-qt
For testing Tkinter/CustomTkinter GUIs.
- **Why**: Simulate user interactions
- **Install**: `pip install pytest-qt`

### 5. pytest-benchmark
For performance testing.
- **Why**: Track performance over time
- **Install**: `pip install pytest-benchmark`

## Setting Up the Testing Environment {#setup}

### Step 1: Install Testing Dependencies
Add to `requirements-dev.txt`:
```
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0
pytest-qt>=4.2.0
pytest-benchmark>=4.0.0
pytest-asyncio>=0.21.0
pytest-timeout>=2.1.0
faker>=19.0.0
hypothesis>=6.80.0
```

Install: `pip install -r requirements-dev.txt`

### Step 2: Create Test Directory Structure
```
tests/
├── __init__.py              # Makes tests a package
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests
│   ├── __init__.py
│   ├── test_services/       # Test service layer
│   │   ├── __init__.py
│   │   ├── test_cache_service.py
│   │   ├── test_file_io_service.py
│   │   └── test_bertopic_service.py
│   ├── test_models/         # Test data models
│   │   ├── __init__.py
│   │   └── test_data_models.py
│   └── test_utils/          # Test utilities
│       ├── __init__.py
│       └── test_logging.py
├── integration/             # Integration tests
│   ├── __init__.py
│   ├── test_data_workflow.py
│   └── test_modeling_workflow.py
├── ui/                      # UI tests
│   ├── __init__.py
│   ├── test_main_window.py
│   └── test_tabs.py
├── fixtures/                # Test data files
│   ├── sample_data.csv
│   ├── sample_data.xlsx
│   └── expected_results.json
└── performance/             # Performance tests
    ├── __init__.py
    └── test_benchmarks.py
```

### Step 3: Configure pytest
Create `pytest.ini` in project root:
```ini
[tool:pytest]
# Test discovery patterns
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Test directories
testpaths = tests

# Output options
addopts = 
    -v                          # Verbose output
    --strict-markers            # Error on unknown markers
    --tb=short                  # Short traceback format
    --cov=src                   # Coverage for src directory
    --cov-report=html           # HTML coverage report
    --cov-report=term-missing   # Terminal report with missing lines

# Markers for categorizing tests
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    ui: marks tests that require UI
    integration: marks integration tests
    unit: marks unit tests
```

## Writing Your First Test {#first-test}

Let's write a simple test for the cache service:

### Step 1: Understand What We're Testing
The cache service generates unique keys for data+model combinations.

### Step 2: Write the Test
```python
# tests/unit/test_services/test_cache_service.py

import pytest
from src.services.cache_service import CacheService
from src.models.data_models import DataConfig, ModelInfo, FileMetadata

class TestCacheService:
    """Test the cache service functionality."""
    
    def test_cache_key_generation_is_deterministic(self):
        """Test that same inputs always generate same cache key."""
        # Arrange (set up test data)
        cache_service = CacheService()
        
        file_metadata = FileMetadata(
            file_path="/path/to/data.csv",
            file_size_bytes=1000,
            row_count=100,
            column_count=5
        )
        
        data_config = DataConfig(
            selected_columns=["text_col"],
            file_metadata=file_metadata
        )
        
        model_info = ModelInfo(
            model_name="test-model",
            model_path="/path/to/model"
        )
        
        # Act (perform the action)
        key1 = cache_service.generate_cache_key(data_config, model_info)
        key2 = cache_service.generate_cache_key(data_config, model_info)
        
        # Assert (check the result)
        assert key1 == key2, "Same inputs should generate same cache key"
        assert len(key1) == 16, "Cache key should be 16 characters"
        assert key1.isalnum(), "Cache key should be alphanumeric"
```

### Understanding the Test Structure: AAA Pattern

1. **Arrange**: Set up test data and environment
2. **Act**: Execute the code being tested
3. **Assert**: Verify the results

## Testing Best Practices {#best-practices}

### 1. Test One Thing at a Time
```python
# Good: Each test has one purpose
def test_cache_key_length():
    assert len(generate_cache_key(...)) == 16

def test_cache_key_uniqueness():
    key1 = generate_cache_key("file1.csv", "model1")
    key2 = generate_cache_key("file2.csv", "model1")
    assert key1 != key2

# Bad: Testing multiple things
def test_cache_key():
    key = generate_cache_key(...)
    assert len(key) == 16  # Testing length
    assert key.isalnum()    # Testing format
    assert key != ""        # Testing not empty
```

### 2. Use Descriptive Test Names
```python
# Good: Clearly describes what's being tested
def test_empty_dataframe_raises_validation_error():
    pass

# Bad: Vague name
def test_validation():
    pass
```

### 3. Use Fixtures for Common Setup
```python
# conftest.py
import pytest

@pytest.fixture
def sample_dataframe():
    """Provide a sample dataframe for tests."""
    return pd.DataFrame({
        'text': ['Document 1', 'Document 2'],
        'label': ['A', 'B']
    })

# In your test
def test_data_validation(sample_dataframe):
    result = validate_data(sample_dataframe)
    assert result.is_valid
```

### 4. Mock External Dependencies
```python
def test_file_loading_handles_missing_file(mocker):
    """Test graceful handling of missing files."""
    # Mock the file system
    mocker.patch('pathlib.Path.exists', return_value=False)
    
    # Test should handle missing file gracefully
    with pytest.raises(FileNotFoundError):
        load_file("missing.csv")
```

### 5. Test Edge Cases
```python
def test_edge_cases():
    # Empty input
    assert process_text("") == ""
    
    # Very long input
    long_text = "x" * 1000000
    result = process_text(long_text)
    assert len(result) <= MAX_LENGTH
    
    # Special characters
    assert process_text("Hello\n\tWorld!@#$") is not None
```

## Running Tests {#running-tests}

### Basic Commands
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_cache_service.py

# Run specific test
pytest tests/unit/test_cache_service.py::test_cache_key_generation

# Run only fast tests
pytest -m "not slow"

# Run with coverage
pytest --cov=src --cov-report=html

# Run in parallel (install pytest-xdist first)
pytest -n auto
```

### Understanding Test Output
```
======================== test session starts ========================
platform linux -- Python 3.9.0, pytest-7.4.0
collected 25 items

tests/unit/test_cache_service.py::test_cache_key_generation PASSED [ 4%]
tests/unit/test_cache_service.py::test_cache_storage PASSED      [ 8%]
tests/unit/test_file_io.py::test_csv_loading PASSED             [12%]
tests/unit/test_file_io.py::test_excel_loading FAILED           [16%]

======================= FAILURES =======================
_____________ test_excel_loading _____________

    def test_excel_loading():
>       data = load_excel("test.xlsx")
E       FileNotFoundError: test.xlsx not found

tests/unit/test_file_io.py:45: FileNotFoundError
=============== 1 failed, 24 passed in 2.34s ===============
```

## Understanding Test Coverage {#coverage}

Coverage shows what percentage of your code is tested.

### Reading Coverage Reports
```
Name                          Stmts   Miss  Cover   Missing
-----------------------------------------------------------
src/services/cache_service.py    45      5    89%   23-27
src/services/file_io.py          78     10    87%   45, 67-75
src/models/data_models.py        34      0   100%
-----------------------------------------------------------
TOTAL                           157     15    90%
```

- **Stmts**: Total statements in file
- **Miss**: Untested statements
- **Cover**: Percentage tested
- **Missing**: Line numbers not tested

### HTML Coverage Report
```bash
# Generate HTML report
pytest --cov=src --cov-report=html

# Open in browser
open htmlcov/index.html
```

## Debugging Failed Tests {#debugging}

### 1. Use pytest's `-s` flag to see print statements
```bash
pytest -s tests/failing_test.py
```

### 2. Use the debugger
```python
def test_something():
    data = process_data()
    import pdb; pdb.set_trace()  # Debugger stops here
    assert data.is_valid
```

### 3. Use pytest's `--pdb` flag
```bash
pytest --pdb  # Drops into debugger on failure
```

### 4. Increase verbosity
```bash
pytest -vv  # Very verbose output
```

## Continuous Integration {#ci}

Set up automated testing with GitHub Actions:

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Next Steps

1. Start with simple unit tests
2. Add integration tests for workflows
3. Add UI tests for critical paths
4. Aim for 80% coverage
5. Make tests part of your development workflow

Remember: Testing is a skill that improves with practice. Start simple and build up!