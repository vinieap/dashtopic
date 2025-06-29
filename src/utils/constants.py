"""
Application constants and configuration values.

This module contains all application-wide constants including file formats,
cache configuration, GUI dimensions, and default algorithm parameters.
"""

from typing import List, Dict, Any

# Application Information
APP_NAME: str = "BERTopic Desktop Application"
APP_VERSION: str = "0.1.0"
APP_DESCRIPTION: str = "Desktop application for simplified BERTopic topic modeling"

# File Formats
SUPPORTED_FORMATS: List[str] = ['.csv', '.xlsx', '.parquet', '.feather']
"""List of supported input file formats for data import."""

EXPORT_FORMATS: List[str] = ['.xlsx', '.csv', '.png', '.svg', '.html']
"""List of supported export formats for results and visualizations."""

# Cache Configuration
CACHE_DIR: str = "cache"
"""Root directory for all cache files."""

EMBEDDINGS_CACHE_DIR: str = "cache/embeddings"
"""Directory for cached document embeddings."""

MODELS_CACHE_DIR: str = "cache/models"
"""Directory for cached model files."""

TEMP_CACHE_DIR: str = "cache/temp"
"""Directory for temporary cache files."""

# GUI Configuration
WINDOW_MIN_WIDTH: int = 1200
"""Minimum window width in pixels."""

WINDOW_MIN_HEIGHT: int = 800
"""Minimum window height in pixels."""

WINDOW_DEFAULT_WIDTH: int = 1400
"""Default window width in pixels."""

WINDOW_DEFAULT_HEIGHT: int = 900
"""Default window height in pixels."""

# Processing Configuration
DEFAULT_BATCH_SIZE: int = 1000
"""Default batch size for processing large datasets."""

MAX_PREVIEW_ROWS: int = 100
"""Maximum number of rows to show in data preview."""

MAX_DOCUMENTS: int = 100000
"""Maximum number of documents to process in a single run."""

# Clustering Algorithms
CLUSTERING_ALGORITHMS: Dict[str, str] = {
    "HDBSCAN": "hdbscan",
    "K-Means": "kmeans", 
    "Agglomerative": "agglomerative",
    "OPTICS": "optics"
}
"""Mapping of user-friendly algorithm names to internal identifiers."""

# Representation Models
REPRESENTATION_MODELS: Dict[str, str] = {
    "KeyBERT": "keybert",
    "Maximal Marginal Relevance": "mmr",
    "Part of Speech": "pos"
}
"""Mapping of representation model names to internal identifiers."""

# UMAP Default Parameters
UMAP_DEFAULTS: Dict[str, Any] = {
    "n_neighbors": 15,
    "n_components": 5,
    "min_dist": 0.0,
    "metric": "cosine",
    "random_state": 42
}
"""Default parameters for UMAP dimensionality reduction."""

# HDBSCAN Default Parameters  
HDBSCAN_DEFAULTS: Dict[str, Any] = {
    "min_cluster_size": 10,
    "min_samples": 5,
    "metric": "euclidean",
    "cluster_selection_method": "eom"
}
"""Default parameters for HDBSCAN clustering algorithm.""" 