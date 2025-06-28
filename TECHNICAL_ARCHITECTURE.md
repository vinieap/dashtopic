# BERTopic Desktop Application - Technical Architecture

## Architecture Overview

The application follows a Model-View-Controller (MVC) pattern with additional service layers for complex operations. The architecture is designed for maintainability, testability, and extensibility.

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                   GUI Layer (View)                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Data Import │ │Topic Modeling│ │Visualization│          │
│  │     Tab     │ │     Tab      │ │     Tab     │   ...    │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                Controller Layer                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │   Main      │ │   Topic     │ │    Data     │          │
│  │Controller   │ │ Controller  │ │ Controller  │   ...    │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                Service Layer                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ BERTopic    │ │ Embedding   │ │   Cache     │          │
│  │  Service    │ │  Service    │ │  Service    │   ...    │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                Data Layer                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ File I/O    │ │   Cache     │ │ Model State │          │
│  │  Manager    │ │  Manager    │ │  Manager    │   ...    │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. GUI Layer (View)

**Technology**: Tkinter with CustomTkinter
**Purpose**: User interface and user interaction handling

#### Main Window Structure
```python
class MainWindow:
    - MenuBar
    - TabWidget
        - DataImportTab
        - ModelConfigTab
        - TopicModelingTab
        - VisualizationTab
        - HyperparameterTab
        - ExportTab
    - StatusBar
    - ProgressDialog
```

#### Key UI Components
- **Custom Widgets**: Parameter input forms, file browsers, progress indicators
- **Plotting Widgets**: Matplotlib/Plotly integration for visualizations
- **Data Tables**: Pandas DataFrame display widgets
- **Configuration Panels**: Dynamic parameter forms based on selected algorithms

### 2. Controller Layer

**Purpose**: Business logic coordination and event handling

#### Main Controller
```python
class MainController:
    def __init__(self):
        self.data_controller = DataController()
        self.topic_controller = TopicController()
        self.visualization_controller = VisualizationController()
        self.export_controller = ExportController()
        
    def coordinate_workflow(self, workflow_type)
    def handle_error(self, error)
    def update_ui_state(self, state)
```

#### Specialized Controllers
- **DataController**: Manages data import, validation, and preprocessing
- **TopicController**: Handles topic modeling operations and parameters
- **VisualizationController**: Manages plot generation and updates
- **ExportController**: Handles all export operations

### 3. Service Layer

**Purpose**: Core business logic and external library integration

#### BERTopic Service
```python
class BERTopicService:
    def configure_model(self, config: ModelConfig) -> BERTopic
    def fit_transform(self, documents: List[str], embeddings: np.ndarray) -> Tuple
    def get_topic_info(self) -> pd.DataFrame
    def visualize_topics(self, **kwargs) -> Figure
    def save_model(self, path: str)
    def load_model(self, path: str) -> BERTopic
```

#### Embedding Service
```python
class EmbeddingService:
    def list_local_models(self) -> List[str]
    def load_model(self, model_name: str) -> SentenceTransformer
    def generate_embeddings(self, texts: List[str]) -> np.ndarray
    def save_embeddings(self, embeddings: np.ndarray, cache_key: str)
    def load_embeddings(self, cache_key: str) -> Optional[np.ndarray]
```

#### Cache Service
```python
class CacheService:
    def generate_cache_key(self, dataset_path: str, columns: List[str]) -> str
    def cache_exists(self, cache_key: str) -> bool
    def save_cache(self, data: Any, cache_key: str)
    def load_cache(self, cache_key: str) -> Any
    def clear_cache(self, pattern: str = None)
    def get_cache_info(self) -> Dict
```

#### Hyperparameter Optimization Service
```python
class OptimizationService:
    def define_search_space(self, algorithm: str) -> Dict
    def run_grid_search(self, config: OptimizationConfig) -> OptimizationResult
    def evaluate_model(self, model: BERTopic, documents: List[str]) -> Dict
    def generate_optimization_plots(self, results: OptimizationResult) -> List[Figure]
```

### 4. Data Layer

**Purpose**: Data persistence, file I/O, and state management

#### File I/O Manager
```python
class FileIOManager:
    def read_dataset(self, file_path: str) -> pd.DataFrame
    def validate_file_format(self, file_path: str) -> bool
    def export_to_excel(self, data: Dict, file_path: str)
    def save_visualization(self, figure: Figure, file_path: str)
    
    # Supported formats
    SUPPORTED_FORMATS = ['.csv', '.xlsx', '.parquet', '.feather']
```

#### Model State Manager
```python
class ModelStateManager:
    def save_application_state(self, state: AppState)
    def load_application_state(self) -> AppState
    def save_model_config(self, config: ModelConfig)
    def load_model_config(self) -> ModelConfig
```

## Data Models

### Configuration Models
```python
@dataclass
class ModelConfig:
    embedding_model: str
    clustering_algorithm: str
    clustering_params: Dict
    umap_params: Dict
    vectorizer_params: Dict
    representation_models: List[str]
    guided_modeling: bool

@dataclass
class DataConfig:
    file_path: str
    selected_columns: List[str]
    text_column: str  # concatenated column name
    preprocessing_options: Dict

@dataclass
class OptimizationConfig:
    search_space: Dict
    cv_folds: int
    scoring_metrics: List[str]
    n_jobs: int
```

### Result Models
```python
@dataclass
class TopicModelResult:
    topics: List[int]
    probabilities: np.ndarray
    topic_info: pd.DataFrame
    topic_words: Dict[int, List[str]]
    model: BERTopic
    
@dataclass
class OptimizationResult:
    best_params: Dict
    best_score: float
    all_results: pd.DataFrame
    score_history: List[float]
```

## Design Patterns

### 1. Observer Pattern
- UI components observe model state changes
- Automatic UI updates when data or configurations change

### 2. Strategy Pattern
- Interchangeable clustering algorithms
- Pluggable representation models
- Configurable evaluation metrics

### 3. Factory Pattern
- Dynamic creation of parameter forms
- Algorithm-specific configuration builders

### 4. Command Pattern
- Undo/redo functionality
- Operation history tracking

## Error Handling Strategy

### Exception Hierarchy
```python
class BERTopicAppException(Exception):
    """Base exception for the application"""

class DataValidationError(BERTopicAppException):
    """Raised when data validation fails"""

class ModelConfigurationError(BERTopicAppException):
    """Raised when model configuration is invalid"""

class CacheError(BERTopicAppException):
    """Raised when cache operations fail"""

class ExportError(BERTopicAppException):
    """Raised when export operations fail"""
```

### Error Handling Flow
1. **Service Layer**: Catch and wrap external library exceptions
2. **Controller Layer**: Handle service exceptions and update UI
3. **View Layer**: Display user-friendly error messages
4. **Logging**: Comprehensive logging for debugging

## Performance Considerations

### Memory Management
- Lazy loading of large datasets
- Chunked processing for large document collections
- Efficient embedding storage and retrieval
- Garbage collection optimization

### Concurrency
- Background threading for long-running operations
- Progress reporting without UI blocking
- Cancellation support for long operations
- Thread-safe caching mechanisms

### Optimization Strategies
- **Embedding Caching**: Avoid recomputation of embeddings
- **Model Persistence**: Save and load trained models
- **Incremental Updates**: Update visualizations incrementally
- **Memory Profiling**: Monitor memory usage patterns

## Testing Strategy

### Unit Tests
- Service layer methods
- Data validation functions
- Cache operations
- Configuration management

### Integration Tests
- Complete workflow testing
- File I/O operations
- Model training and prediction
- Export functionality

### UI Tests
- User interaction scenarios (using tkinter testing utilities)
- Error handling in UI
- Performance under load
- Cross-platform compatibility

## Deployment Architecture

### Package Structure
```
bertopic_app/
├── main.py                 # Application entry point
├── gui/                    # GUI components
│   ├── __init__.py
│   ├── main_window.py
│   ├── tabs/
│   └── widgets/
├── controllers/            # Controller layer
├── services/              # Business logic services
├── models/                # Data models and configurations
├── utils/                 # Utility functions
├── resources/             # UI resources, icons, etc.
├── tests/                 # Test suite
└── requirements.txt       # Python dependencies
```

### Distribution
- **PyInstaller**: Create standalone executable
- **NSIS**: Windows installer creation
- **Code Signing**: Digital signature for trust
- **Auto-updater**: Future update mechanism

## Security Considerations

### Input Validation
- File path sanitization to prevent directory traversal
- Parameter validation to prevent injection
- File format verification before processing
- Data type validation for all inputs

### Data Protection
- Local-only processing (no network calls)
- Secure temporary file handling
- Cache encryption for sensitive data
- Memory clearing after processing 