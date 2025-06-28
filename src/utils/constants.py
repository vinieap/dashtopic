"""
Application constants and configuration values.
"""

# Application Information
APP_NAME = "BERTopic Desktop Application"
APP_VERSION = "0.1.0"
APP_DESCRIPTION = "Desktop application for simplified BERTopic topic modeling"

# File Formats
SUPPORTED_FORMATS = ['.csv', '.xlsx', '.parquet', '.feather']
EXPORT_FORMATS = ['.xlsx', '.csv', '.png', '.svg', '.html']

# Cache Configuration
CACHE_DIR = "cache"
EMBEDDINGS_CACHE_DIR = "cache/embeddings"
MODELS_CACHE_DIR = "cache/models"
TEMP_CACHE_DIR = "cache/temp"

# GUI Configuration
WINDOW_MIN_WIDTH = 1200
WINDOW_MIN_HEIGHT = 800
WINDOW_DEFAULT_WIDTH = 1400
WINDOW_DEFAULT_HEIGHT = 900

# Processing Configuration
DEFAULT_BATCH_SIZE = 1000
MAX_PREVIEW_ROWS = 100
MAX_DOCUMENTS = 100000

# Clustering Algorithms
CLUSTERING_ALGORITHMS = {
    "HDBSCAN": "hdbscan",
    "K-Means": "kmeans", 
    "Agglomerative": "agglomerative",
    "OPTICS": "optics"
}

# Representation Models
REPRESENTATION_MODELS = {
    "KeyBERT": "keybert",
    "Maximal Marginal Relevance": "mmr",
    "Part of Speech": "pos"
}

# UMAP Default Parameters
UMAP_DEFAULTS = {
    "n_neighbors": 15,
    "n_components": 5,
    "min_dist": 0.0,
    "metric": "cosine",
    "random_state": 42
}

# HDBSCAN Default Parameters  
HDBSCAN_DEFAULTS = {
    "min_cluster_size": 10,
    "min_samples": 5,
    "metric": "euclidean",
    "cluster_selection_method": "eom"
} 