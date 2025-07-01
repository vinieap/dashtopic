# BERTopic Desktop Application - Project Requirements

## Overview
A comprehensive Windows desktop application built with Python to simplify and extend BERTopic functionality with an intuitive GUI interface. The application operates completely offline without internet connectivity.

## Core Features

### 1. Data Management
- **Dataset Import**: Support for multiple file formats (CSV, Excel, Parquet, Feather)
- **Column Selection**: Multi-column selection with automatic space concatenation
- **Data Preview**: Display sample data before processing
- **Data Validation**: Check for empty values, data types, and format consistency

### 2. Embedding Management
- **Local Models Only**: Use SentenceTransformers models stored locally
- **Model Selection**: Browse and select from available local models
- **Embedding Caching**: Generate and save embeddings with intelligent naming (dataset + columns)
- **Cache Management**: Automatic loading of existing embeddings, cache cleanup options

### 3. Clustering Configuration
- **Algorithm Selection**: Support for multiple clustering algorithms
  - HDBSCAN (default)
  - K-Means
  - AgglomerativeClustering
  - OPTICS
- **Parameter Configuration**: Dynamic parameter forms for each algorithm
- **Advanced Options**: Custom clustering parameter combinations

### 4. Dimensionality Reduction
- **UMAP Integration**: For both visualization and topic modeling
- **Parameter Control**: Configurable UMAP parameters
  - n_neighbors
  - n_components
  - min_dist
  - metric
  - random_state
- **Visualization Options**: 2D and 3D scatter plots

### 5. Topic Modeling Features
- **Vectorization Options**: TF-IDF, Count Vectorizer with parameter control
- **c-TF-IDF Models**: Configurable parameters and options
- **Guided Modeling**: Force c-TF-IDF usage for guided topic discovery
- **Representation Models**: Support for multiple representation model types
  - KeyBERT
  - MaximalMarginalRelevance
  - PartOfSpeech
  - Custom models

### 6. Hyperparameter Optimization
- **Grid Search**: Automated parameter optimization
- **Custom Metrics**: Multiple evaluation metrics for model assessment
- **Results Visualization**: Interactive charts showing parameter performance
- **Export Results**: Save optimization results and best parameters

### 7. Visualization and Export
- **Topic Visualization**: Interactive plots showing topic distributions
- **Word Clouds**: Per-topic word importance visualization
- **N-gram Analysis**: Top-n n-gram words per topic (configurable n)
- **Export Options**: 
  - Excel spreadsheets with results
  - PNG/SVG visualizations
  - Model artifacts

## Technical Requirements

### Platform
- **Target OS**: Windows 10/11
- **Python Version**: 3.8+
- **GUI Framework**: Tkinter with CustomTkinter
- **Packaging**: PyInstaller for standalone executable

### Dependencies
- BERTopic
- SentenceTransformers
- UMAP-learn
- scikit-learn
- pandas
- numpy
- matplotlib
- plotly
- openpyxl
- pyarrow (for Parquet support)
- customtkinter
- pillow (for CustomTkinter image support)

### Performance Requirements
- Handle datasets up to 100,000 documents efficiently
- Responsive UI during long-running operations
- Progress indicators for all major operations
- Memory-efficient processing for large datasets

### User Experience Requirements
- Intuitive tabbed interface
- Real-time parameter validation
- Helpful tooltips and documentation
- Error handling with user-friendly messages
- Undo/redo capabilities where applicable

## Application Structure

### Main Tabs
1. **Data Import**: Dataset loading and column selection
2. **Model Configuration**: Embedding models and parameters
3. **Topic Modeling**: Main BERTopic configuration and execution
4. **Visualization**: Interactive plots and charts
5. **Hyperparameter Tuning**: Automated optimization
6. **Export**: Results export and visualization saving

### Secondary Features
- **Settings**: Application preferences and cache management
- **Help**: Built-in documentation and tutorials
- **About**: Version information and credits

## Data Flow
1. Import dataset → Validate data
2. Select columns → Concatenate text features
3. Check for cached embeddings → Generate if needed
4. Configure clustering parameters → Set UMAP parameters
5. Run BERTopic model → Generate topics
6. Visualize results → Export findings
7. Optional: Run hyperparameter optimization

## Security and Privacy
- **Offline Operation**: No internet connectivity required
- **Local Data**: All processing done locally
- **Cache Security**: Secure storage of cached embeddings
- **Data Validation**: Input sanitization for file paths and parameters

## Future Enhancements
- **Batch Processing**: Process multiple datasets
- **Model Comparison**: Side-by-side comparison of different configurations
- **Advanced Visualization**: 3D interactive topic spaces
- **Plugin Architecture**: Support for custom representation models
- **Cloud Sync**: Optional cloud backup of results (future version) 