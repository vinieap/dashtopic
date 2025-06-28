# BERTopic Desktop Application

A comprehensive Windows desktop application for simplified and extended BERTopic topic modeling with an intuitive GUI interface. This application operates completely offline without requiring internet connectivity.

## 🎯 Project Overview

This project aims to create a user-friendly desktop application that makes BERTopic accessible to researchers, analysts, and data scientists who need powerful topic modeling capabilities without the complexity of coding. The application provides a complete workflow from data import to results export with advanced visualization and hyperparameter optimization features.

## ✨ Key Features

### 🔄 Complete Workflow
- **Data Import**: Support for CSV, Excel, Parquet, and Feather formats
- **Column Selection**: Multi-column text concatenation with preview
- **Embedding Generation**: Local SentenceTransformers models with intelligent caching
- **Topic Modeling**: BERTopic integration with multiple clustering algorithms
- **Visualization**: Interactive 2D/3D plots, word clouds, and topic analysis
- **Export**: Excel, visualizations, and HTML dashboard generation

### 🤖 Advanced Modeling
- **Multiple Clustering Algorithms**: HDBSCAN, K-Means, Agglomerative, OPTICS
- **UMAP Integration**: Configurable dimensionality reduction for modeling and visualization
- **Vectorization Options**: TF-IDF and Count vectorizers with full parameter control
- **Representation Models**: KeyBERT, Maximal Marginal Relevance, Part-of-Speech
- **Guided Modeling**: Seed topic input with c-TF-IDF enforcement

### 🔍 Optimization & Analysis
- **Hyperparameter Tuning**: Automated grid search with multiple evaluation metrics
- **Performance Visualization**: Parameter importance and correlation analysis
- **N-gram Analysis**: Configurable n-gram extraction and visualization
- **Interactive Exploration**: Click-to-explore topic details and document selection

### 🚀 Performance & Usability
- **Intelligent Caching**: Automatic embedding caching based on dataset and model
- **Memory Optimization**: Efficient processing for large datasets (up to 100K documents)
- **Progress Tracking**: Real-time progress indicators with cancellation support
- **Modern UI**: CustomTkinter provides a sleek, modern interface with smooth animations
- **Lightweight**: Built on Tkinter for fast startup and low resource usage
- **Cross-Platform**: Native appearance on Windows, macOS, and Linux

## 📋 Planning Documents

This repository contains comprehensive planning documentation created before implementation:

### 📄 Core Documentation
- **[PROJECT_REQUIREMENTS.md](PROJECT_REQUIREMENTS.md)**: Detailed feature specifications and technical requirements
- **[TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md)**: System architecture, design patterns, and component structure
- **[UI_UX_DESIGN.md](UI_UX_DESIGN.md)**: Complete user interface design specifications with mockups
- **[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md)**: 16-week development timeline with phases and milestones
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: Recommended directory organization and file structure

### 🏗️ Architecture Diagrams
The project includes visual architecture diagrams showing:
- **System Architecture**: MVC pattern with service layers
- **Data Flow**: Complete pipeline from input to output
- **Component Relationships**: How different parts interact

## 🛠️ Technology Stack

### Core Technologies
- **GUI Framework**: Tkinter with CustomTkinter for modern styling
- **Topic Modeling**: BERTopic with SentenceTransformers
- **Clustering**: HDBSCAN, K-Means, scikit-learn algorithms
- **Dimensionality Reduction**: UMAP
- **Visualization**: Matplotlib and Plotly for interactive plots
- **Data Processing**: Pandas, NumPy

### Development Tools
- **Language**: Python 3.8+
- **Testing**: pytest with Tkinter testing utilities
- **Code Quality**: Black, flake8, mypy
- **Packaging**: PyInstaller for standalone executables
- **CI/CD**: GitHub Actions for automated testing and releases

## 📊 Application Architecture

The application follows a layered MVC architecture:

```
┌─────────────────────────────────────────┐
│               GUI Layer                 │ ← User Interface (Tkinter + CustomTkinter)
│     (Tabs, Widgets, Dialogs)          │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│            Controller Layer             │ ← Business Logic Coordination
│   (Data, Topic, Visualization, Export) │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│             Service Layer               │ ← Core Business Logic
│  (BERTopic, Embedding, Cache, Export)  │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│              Data Layer                 │ ← Persistence & I/O
│    (File I/O, Cache, Model State)      │
└─────────────────────────────────────────┘
```

## 🚀 Getting Started (Future Implementation)

### Prerequisites
- Windows 10/11
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/bertopic-desktop.git

# Navigate to project directory
cd bertopic-desktop

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Building from Source
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Build executable
python scripts/build.py

# Package for distribution
python scripts/package.py
```

## 📈 Development Timeline

The project is planned for **12-16 weeks** of development across 6 phases:

1. **Foundation Setup** (Weeks 1-2): Project structure and basic GUI
2. **Data Management** (Weeks 3-4): File I/O and data validation
3. **Embedding & Caching** (Weeks 5-6): Model integration and caching system
4. **Core Topic Modeling** (Weeks 7-9): BERTopic integration and algorithms
5. **Visualization & Polish** (Weeks 10-12): Interactive plots and UI refinement
6. **Optimization & Release** (Weeks 13-16): Hyperparameter tuning and deployment

## 🎨 User Interface Preview

The application features a modern tabbed interface:

- **Data Import Tab**: File loading and column selection
- **Model Configuration Tab**: Embedding models and clustering setup
- **Topic Modeling Tab**: Main modeling interface with progress tracking
- **Visualization Tab**: Interactive plots and topic exploration
- **Hyperparameter Tab**: Automated parameter optimization
- **Export Tab**: Results export and visualization saving

## 🔒 Security & Privacy

- **Offline Operation**: Complete functionality without internet access
- **Local Processing**: All data stays on the user's machine
- **Secure Caching**: Encrypted cache storage for sensitive data
- **Input Validation**: Comprehensive validation to prevent security issues

## 📝 Documentation Standards

All code will follow these standards:
- **Type Hints**: Complete type annotations for better IDE support
- **Docstrings**: Comprehensive documentation for all public methods
- **Code Comments**: Clear explanations for complex logic
- **User Manual**: Step-by-step usage instructions
- **API Reference**: Complete developer documentation

## 🧪 Testing Strategy

Comprehensive testing approach:
- **Unit Tests**: >90% code coverage target
- **Integration Tests**: Full workflow testing
- **UI Tests**: Automated GUI interaction testing
- **Performance Tests**: Memory and speed benchmarks

## 🤝 Contributing (Future)

When the project moves to implementation:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📄 License

This project will be released under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **BERTopic**: Maarten Grootendorst's excellent topic modeling library
- **SentenceTransformers**: Hugging Face's sentence embedding library
- **UMAP**: McInnes et al.'s dimensionality reduction algorithm
- **CustomTkinter**: Tom Schimansky's modern Tkinter UI library