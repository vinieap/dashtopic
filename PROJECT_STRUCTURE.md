# BERTopic Desktop Application - Project Structure

## Directory Organization

```
bertopic_app/
├── 📁 .github/                     # GitHub workflows and templates
│   ├── workflows/
│   │   ├── ci.yml                  # Continuous integration
│   │   └── release.yml             # Release automation
│   └── ISSUE_TEMPLATE.md           # Issue template
│
├── 📁 docs/                        # Documentation
│   ├── PROJECT_REQUIREMENTS.md    # Project requirements (this file)
│   ├── TECHNICAL_ARCHITECTURE.md  # Technical architecture
│   ├── UI_UX_DESIGN.md           # UI/UX specifications
│   ├── IMPLEMENTATION_ROADMAP.md  # Development roadmap
│   ├── USER_MANUAL.md             # User documentation
│   ├── API_REFERENCE.md           # API documentation
│   └── 📁 images/                 # Documentation images
│
├── 📁 src/                        # Source code
│   ├── __init__.py
│   ├── main.py                    # Application entry point
│   │
│   ├── 📁 gui/                    # User interface components
│   │   ├── __init__.py
│   │   ├── main_window.py         # Main application window
│   │   ├── 📁 tabs/               # Tab implementations
│   │   │   ├── __init__.py
│   │   │   ├── data_import_tab.py
│   │   │   ├── model_config_tab.py
│   │   │   ├── topic_modeling_tab.py
│   │   │   ├── visualization_tab.py
│   │   │   ├── hyperparameter_tab.py
│   │   │   └── export_tab.py
│   │   ├── 📁 widgets/            # Custom widgets
│   │   │   ├── __init__.py
│   │   │   ├── parameter_form.py  # Dynamic parameter forms
│   │   │   ├── data_table.py      # Data display tables
│   │   │   ├── plot_widget.py     # Plotting integration
│   │   │   ├── progress_dialog.py # Progress indicators
│   │   │   └── file_browser.py    # File selection widget
│   │   └── 📁 dialogs/            # Dialog windows
│   │       ├── __init__.py
│   │       ├── settings_dialog.py
│   │       ├── about_dialog.py
│   │       └── error_dialog.py
│   │
│   ├── 📁 controllers/            # Business logic controllers
│   │   ├── __init__.py
│   │   ├── main_controller.py     # Central coordinator
│   │   ├── data_controller.py     # Data management
│   │   ├── topic_controller.py    # Topic modeling operations
│   │   ├── visualization_controller.py  # Visualization management
│   │   └── export_controller.py   # Export operations
│   │
│   ├── 📁 services/               # Core business services
│   │   ├── __init__.py
│   │   ├── bertopic_service.py    # BERTopic integration
│   │   ├── embedding_service.py   # Embedding generation
│   │   ├── cache_service.py       # Caching system
│   │   ├── optimization_service.py # Hyperparameter optimization
│   │   ├── visualization_service.py # Plot generation
│   │   └── export_service.py      # Data export functionality
│   │
│   ├── 📁 models/                 # Data models and configurations
│   │   ├── __init__.py
│   │   ├── app_config.py          # Application configuration
│   │   ├── data_models.py         # Data structure definitions
│   │   ├── model_config.py        # ML model configurations
│   │   ├── optimization_config.py # Optimization settings
│   │   └── export_config.py       # Export configurations
│   │
│   ├── 📁 utils/                  # Utility functions
│   │   ├── __init__.py
│   │   ├── file_utils.py          # File operations
│   │   ├── validation_utils.py    # Data validation
│   │   ├── cache_utils.py         # Cache management utilities
│   │   ├── logging_utils.py       # Logging configuration
│   │   ├── error_handling.py      # Error handling utilities
│   │   └── constants.py           # Application constants
│   │
│   └── 📁 data/                   # Data layer
│       ├── __init__.py
│       ├── file_io_manager.py     # File I/O operations
│       ├── cache_manager.py       # Cache storage management
│       └── model_state_manager.py # Model persistence
│
├── 📁 resources/                  # Application resources
│   ├── 📁 icons/                  # Application icons
│   │   ├── app_icon.ico
│   │   ├── toolbar_icons/
│   │   └── status_icons/
│   ├── 📁 styles/                 # UI stylesheets
│   │   ├── main_style.qss
│   │   └── dark_theme.qss
│   ├── 📁 fonts/                  # Custom fonts (if needed)
│   └── 📁 templates/              # Export templates
│       ├── excel_template.xlsx
│       └── html_dashboard.html
│
├── 📁 tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py                # Pytest configuration
│   ├── 📁 unit/                   # Unit tests
│   │   ├── test_services/
│   │   ├── test_controllers/
│   │   ├── test_models/
│   │   └── test_utils/
│   ├── 📁 integration/            # Integration tests
│   │   ├── test_workflows/
│   │   ├── test_data_pipeline/
│   │   └── test_export/
│   ├── 📁 ui/                     # UI tests
│   │   ├── test_main_window.py
│   │   └── test_tabs/
│   └── 📁 fixtures/               # Test data
│       ├── sample_data.csv
│       ├── test_embeddings.pkl
│       └── mock_models/
│
├── 📁 scripts/                    # Build and deployment scripts
│   ├── build.py                   # Build script
│   ├── package.py                 # Packaging script
│   ├── install_dependencies.py   # Dependency installation
│   └── release.py                 # Release automation
│
├── 📁 cache/                      # Application cache (created at runtime)
│   ├── embeddings/                # Cached embeddings
│   ├── models/                    # Cached models
│   └── temp/                      # Temporary files
│
├── 📁 logs/                       # Application logs (created at runtime)
│   ├── app.log                    # Main application log
│   ├── error.log                  # Error log
│   └── performance.log            # Performance metrics
│
├── 📄 main.py                     # Application entry point
├── 📄 requirements.txt            # Python dependencies
├── 📄 requirements-dev.txt        # Development dependencies
├── 📄 setup.py                    # Package setup
├── 📄 pyproject.toml              # Modern Python packaging
├── 📄 .gitignore                  # Git ignore rules
├── 📄 .env.example                # Environment variables template
├── 📄 README.md                   # Project overview
├── 📄 CHANGELOG.md                # Version history
├── 📄 LICENSE                     # Software license
└── 📄 VERSION                     # Version file
```

## File Descriptions

### Core Application Files

#### `main.py`
Application entry point and bootstrap code.
```python
"""
BERTopic Desktop Application
Main entry point for the application.
"""
import sys
import logging
import customtkinter as ctk
from src.gui.main_window import MainWindow
from src.utils.logging_utils import setup_logging

def main():
    setup_logging()
    
    # Set CustomTkinter appearance mode and color theme
    ctk.set_appearance_mode("System")  # Modes: "System", "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"
    
    # Create and run the application
    app = MainWindow()
    app.mainloop()

if __name__ == "__main__":
    main()
```

#### `requirements.txt`
Core Python dependencies.
```
bertopic>=0.15.0
sentence-transformers>=2.2.0
umap-learn>=0.5.3
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
plotly>=5.15.0
customtkinter>=5.2.0
pillow>=10.0.0
openpyxl>=3.1.0
pyarrow>=12.0.0
```

#### `requirements-dev.txt`
Development dependencies.
```
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-xvfb>=3.0.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
pre-commit>=3.3.0
sphinx>=7.1.0
```

### GUI Layer Structure

#### `gui/main_window.py`
Main application window with menu bar, toolbar, tab widget, and status bar.

#### `gui/tabs/`
Individual tab implementations:
- **data_import_tab.py**: File loading and column selection
- **model_config_tab.py**: Embedding model and clustering configuration
- **topic_modeling_tab.py**: Main topic modeling interface
- **visualization_tab.py**: Interactive plots and visualizations
- **hyperparameter_tab.py**: Parameter optimization interface
- **export_tab.py**: Data and visualization export

#### `gui/widgets/`
Reusable custom widgets:
- **parameter_form.py**: Dynamic forms for algorithm parameters
- **data_table.py**: Enhanced table widget for data display
- **plot_widget.py**: Matplotlib/Plotly integration widget
- **progress_dialog.py**: Progress tracking with cancellation
- **file_browser.py**: Enhanced file selection with preview

### Business Logic Layer

#### `controllers/`
MVC controllers coordinating between GUI and services:
- **main_controller.py**: Central application coordinator
- **data_controller.py**: Data import and validation logic
- **topic_controller.py**: Topic modeling workflow management
- **visualization_controller.py**: Plot generation coordination
- **export_controller.py**: Export operation management

#### `services/`
Core business logic services:
- **bertopic_service.py**: BERTopic model integration
- **embedding_service.py**: SentenceTransformers integration
- **cache_service.py**: Intelligent caching system
- **optimization_service.py**: Hyperparameter optimization
- **visualization_service.py**: Plot generation and management
- **export_service.py**: Data export in multiple formats

### Data Layer

#### `models/`
Data models and configuration classes:
- **app_config.py**: Application-wide configuration
- **data_models.py**: Dataset and processing configurations
- **model_config.py**: ML model parameters and settings
- **optimization_config.py**: Hyperparameter optimization settings
- **export_config.py**: Export format and option configurations

#### `data/`
Data persistence and management:
- **file_io_manager.py**: File reading/writing operations
- **cache_manager.py**: Cache storage and retrieval
- **model_state_manager.py**: Model persistence and state management

### Utility Layer

#### `utils/`
Shared utility functions:
- **file_utils.py**: File operation helpers
- **validation_utils.py**: Data validation functions
- **cache_utils.py**: Cache management utilities
- **logging_utils.py**: Logging configuration and helpers
- **error_handling.py**: Exception handling utilities
- **constants.py**: Application-wide constants

## Configuration Files

### `pyproject.toml`
Modern Python packaging configuration:
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "bertopic-desktop"
version = "1.0.0"
description = "Desktop application for BERTopic topic modeling"
authors = [{name = "Your Name", email = "your.email@example.com"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.8"

[project.scripts]
bertopic-desktop = "src.main:main"

[tool.black]
line-length = 88
target-version = ['py38']

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
```

### `.gitignore`
Git ignore patterns:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Application specific
cache/
logs/
*.pkl
*.model
temp/

# OS specific
.DS_Store
Thumbs.db
```

## Testing Structure

### Unit Tests
```
tests/unit/
├── test_services/
│   ├── test_bertopic_service.py
│   ├── test_embedding_service.py
│   └── test_cache_service.py
├── test_controllers/
├── test_models/
└── test_utils/
```

### Integration Tests
```
tests/integration/
├── test_workflows/
│   ├── test_full_pipeline.py
│   └── test_optimization_workflow.py
├── test_data_pipeline/
└── test_export/
```

### UI Tests
```
tests/ui/
├── test_main_window.py
├── test_tabs/
│   ├── test_data_import_tab.py
│   └── test_topic_modeling_tab.py
└── test_widgets/
```

## Resource Organization

### Icons
- **app_icon.ico**: Main application icon
- **toolbar_icons/**: Toolbar button icons
- **status_icons/**: Status indicator icons

### Styles
- **main_style.qss**: Main application stylesheet
- **dark_theme.qss**: Dark theme stylesheet

### Templates
- **excel_template.xlsx**: Excel export template
- **html_dashboard.html**: HTML dashboard template

## Build and Deployment

### Build Scripts
- **build.py**: Main build script
- **package.py**: PyInstaller packaging
- **install_dependencies.py**: Automated dependency installation
- **release.py**: Release automation and distribution

### GitHub Workflows
- **ci.yml**: Continuous integration testing
- **release.yml**: Automated release creation

## Development Workflow

1. **Setup**: Clone repository, create virtual environment, install dependencies
2. **Development**: Follow TDD approach with unit tests
3. **Testing**: Run full test suite before commits
4. **Code Quality**: Use pre-commit hooks for formatting and linting
5. **Documentation**: Update documentation with changes
6. **Release**: Use automated release pipeline

## Security Considerations

### File Path Handling
- All file paths are validated and sanitized
- No direct file path input from users without validation
- Use Path objects for cross-platform compatibility

### Cache Security
- Cache files are stored in user-specific directories
- Cache integrity validation before loading
- Automatic cache cleanup for security

### Error Handling
- No sensitive information in error messages
- Comprehensive logging without exposing internals
- Graceful degradation for missing dependencies

This project structure provides a solid foundation for building a maintainable, scalable, and professional BERTopic desktop application. 