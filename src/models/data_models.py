"""
Data models for file handling and validation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class FileMetadata:
    """Metadata information about a loaded file."""

    file_path: str
    file_name: str
    file_size_bytes: int
    file_format: str
    row_count: int
    column_count: int
    columns: List[str]
    data_types: Dict[str, str]
    encoding: Optional[str] = None
    has_header: bool = True
    delimiter: Optional[str] = None
    preview_data: Optional[pd.DataFrame] = None
    
    # Streaming and memory optimization attributes
    supports_streaming: bool = False
    is_estimate: bool = False  # True if row_count is estimated
    is_sample: bool = False  # True if this metadata is from a sample
    sample_size: Optional[int] = None  # Size of sample if is_sample=True

    @property
    def file_size_mb(self) -> float:
        """File size in megabytes."""
        return self.file_size_bytes / (1024 * 1024)

    @property
    def text_columns(self) -> List[str]:
        """Get list of columns that contain text data."""
        text_types = ["object", "string"]
        return [
            col
            for col, dtype in self.data_types.items()
            if any(t in str(dtype).lower() for t in text_types)
        ]


@dataclass
class ValidationResult:
    """Results from data validation checks."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    column_info: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)

    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)

    @property
    def has_issues(self) -> bool:
        """Check if there are any errors or warnings."""
        return len(self.errors) > 0 or len(self.warnings) > 0


@dataclass
class DataConfig:
    """Configuration for data processing and analysis."""

    file_metadata: Optional[FileMetadata] = None
    selected_columns: List[str] = field(default_factory=list)
    text_combination_method: str = (
        "concatenate"  # "concatenate", "join_spaces", "join_newlines"
    )
    text_combination_separator: str = " "
    include_column_names: bool = False
    remove_empty_rows: bool = True
    min_text_length: int = 10
    max_text_length: Optional[int] = None
    max_documents: Optional[int] = None  # Limit number of documents to process
    preprocessing_steps: List[str] = field(default_factory=list)

    @property
    def is_configured(self) -> bool:
        """Check if data is properly configured for analysis."""
        return (
            self.file_metadata is not None
            and len(self.selected_columns) > 0
            and all(col in self.file_metadata.columns for col in self.selected_columns)
        )

    @property
    def combined_text_preview(self) -> str:
        """Generate a preview of how text will be combined."""
        if (
            not self.is_configured
            or self.file_metadata.preview_data is None
            or self.file_metadata.preview_data.empty
        ):
            return "No data available for preview"

        try:
            # Get first few rows for preview
            preview_df = self.file_metadata.preview_data.head(3)
            combined_texts = []

            for _, row in preview_df.iterrows():
                texts = []
                for col in self.selected_columns:
                    if col in row and pd.notna(row[col]):
                        text = str(row[col]).strip()
                        if text:
                            if self.include_column_names:
                                texts.append(f"{col}: {text}")
                            else:
                                texts.append(text)

                if texts:
                    combined_text = self.text_combination_separator.join(texts)
                    combined_texts.append(
                        combined_text[:200] + "..."
                        if len(combined_text) > 200
                        else combined_text
                    )

            return "\n\n".join(
                f"Row {i+1}: {text}" for i, text in enumerate(combined_texts)
            )

        except Exception as e:
            return f"Error generating preview: {str(e)}"


@dataclass
class DataQualityMetrics:
    """Metrics for assessing data quality."""

    total_rows: int
    empty_rows: int
    duplicate_rows: int
    columns_with_missing: Dict[str, int]
    text_length_stats: Dict[str, float]  # min, max, mean, median
    character_encoding_issues: int
    data_completeness: float  # percentage of non-empty cells

    @property
    def completeness_percentage(self) -> float:
        """Data completeness as percentage."""
        return round(self.data_completeness * 100, 2)


# Phase 3: Embedding and Model Management Models


@dataclass
class ModelInfo:
    """Information about an available embedding model."""

    model_name: str
    model_path: str
    model_type: str  # "sentence-transformers", "huggingface", "local"
    embedding_dimension: Optional[int] = None
    max_sequence_length: Optional[int] = None
    description: Optional[str] = None
    model_size_mb: Optional[float] = None
    is_loaded: bool = False
    load_time_seconds: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    supports_languages: List[str] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        """User-friendly display name for the model."""
        return self.model_name.replace("_", " ").replace("-", " ").title()

    @property
    def is_local(self) -> bool:
        """Check if model is stored locally."""
        return Path(self.model_path).exists() if self.model_path else False


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model_info: Optional[ModelInfo] = None
    batch_size: int = 32
    max_length: Optional[int] = None
    normalize_embeddings: bool = True
    convert_to_tensor: bool = True
    device: str = "auto"  # "auto", "cpu", "cuda"
    show_progress_bar: bool = True
    num_workers: int = 0

    @property
    def is_configured(self) -> bool:
        """Check if embedding configuration is valid."""
        import logging
        logger = logging.getLogger(__name__)
        
        if self.model_info is None:
            logger.debug("EmbeddingConfig.is_configured: model_info is None")
            return False
        
        is_loaded = self.model_info.is_loaded
        logger.debug(f"EmbeddingConfig.is_configured: model_info.is_loaded = {is_loaded}")
        logger.debug(f"EmbeddingConfig.is_configured: model_info.model_type = {self.model_info.model_type}")
        logger.debug(f"EmbeddingConfig.is_configured: model_info.model_name = {self.model_info.model_name}")
        
        return is_loaded


@dataclass
class CacheInfo:
    """Information about cached embeddings."""

    cache_key: str
    file_path: str
    creation_time: datetime
    last_accessed: datetime
    file_size_bytes: int
    embedding_shape: Tuple[int, int]  # (num_documents, embedding_dim)
    model_name: str
    data_hash: str
    is_valid: bool = True

    @property
    def file_size_mb(self) -> float:
        """Cache file size in megabytes."""
        return self.file_size_bytes / (1024 * 1024)

    @property
    def age_hours(self) -> float:
        """Age of cache in hours."""
        return (datetime.now() - self.creation_time).total_seconds() / 3600


@dataclass
class EmbeddingResult:
    """Result of embedding generation process."""

    embeddings: Optional[np.ndarray] = None
    texts: List[str] = field(default_factory=list)
    model_info: Optional[ModelInfo] = None
    processing_time_seconds: float = 0.0
    cache_hit: bool = False
    cache_info: Optional[CacheInfo] = None
    batch_stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_documents(self) -> int:
        """Number of documents processed."""
        return len(self.texts)

    @property
    def embedding_dimension(self) -> int:
        """Dimension of embeddings."""
        return self.embeddings.shape[1] if self.embeddings is not None else 0

    @property
    def memory_usage_mb(self) -> float:
        """Estimated memory usage in MB."""
        if self.embeddings is None:
            return 0.0
        return (self.embeddings.nbytes) / (1024 * 1024)


# Phase 4: Topic Modeling Data Models


@dataclass
class ClusteringConfig:
    """Configuration for clustering algorithms."""

    algorithm: str = "hdbscan"  # "hdbscan", "kmeans", "agglomerative", "optics"

    # HDBSCAN parameters
    min_cluster_size: int = 10
    min_samples: Optional[int] = None
    cluster_selection_epsilon: float = 0.0
    max_cluster_size: Optional[int] = None
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"  # "eom", "leaf"
    prediction_data: bool = True

    # K-Means parameters
    n_clusters: int = 8
    init: str = "k-means++"
    n_init: int = 10
    max_iter: int = 300
    random_state: Optional[int] = 42

    # Agglomerative parameters
    linkage: str = "ward"  # "ward", "complete", "average", "single"
    affinity: str = "euclidean"  # "euclidean", "cosine", "manhattan"

    # OPTICS parameters
    min_samples_optics: int = 5
    max_eps: float = float("inf")
    xi: float = 0.05
    min_cluster_size_optics: Optional[int] = None

    def get_algorithm_params(self) -> Dict[str, Any]:
        """Get parameters for the selected algorithm."""
        if self.algorithm == "hdbscan":
            params = {
                "min_cluster_size": self.min_cluster_size,
                "metric": self.metric,
                "cluster_selection_method": self.cluster_selection_method,
                "prediction_data": self.prediction_data,
            }
            if self.min_samples is not None:
                params["min_samples"] = self.min_samples
            if self.cluster_selection_epsilon > 0:
                params["cluster_selection_epsilon"] = self.cluster_selection_epsilon
            if self.max_cluster_size is not None:
                params["max_cluster_size"] = self.max_cluster_size
            return params

        elif self.algorithm == "kmeans":
            return {
                "n_clusters": self.n_clusters,
                "init": self.init,
                "n_init": self.n_init,
                "max_iter": self.max_iter,
                "random_state": self.random_state,
            }

        elif self.algorithm == "agglomerative":
            return {
                "n_clusters": self.n_clusters,
                "linkage": self.linkage,
                "affinity": self.affinity,
            }

        elif self.algorithm == "optics":
            params = {
                "min_samples": self.min_samples_optics,
                "max_eps": self.max_eps,
                "xi": self.xi,
                "metric": self.metric,
            }
            if self.min_cluster_size_optics is not None:
                params["min_cluster_size"] = self.min_cluster_size_optics
            return params

        return {}


@dataclass
class VectorizationConfig:
    """Configuration for vectorization methods."""

    vectorizer_type: str = "tfidf"  # "tfidf", "count"

    # Common parameters
    max_features: Optional[int] = None
    min_df: Union[int, float] = 1
    max_df: Union[int, float] = 1.0
    ngram_range: Tuple[int, int] = (1, 1)
    stop_words: Optional[str] = "english"  # "english", None, or custom list
    lowercase: bool = True
    max_df_ratio: float = 0.95  # Remove words that appear in more than X% of documents
    min_df_count: int = 2  # Remove words that appear in less than X documents

    # TF-IDF specific
    use_idf: bool = True
    smooth_idf: bool = True
    sublinear_tf: bool = False
    norm: Optional[str] = "l2"  # "l1", "l2", None

    # Count specific
    binary: bool = False

    def get_vectorizer_params(self) -> Dict[str, Any]:
        """Get parameters for the selected vectorizer."""
        base_params = {
            "max_features": self.max_features,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "ngram_range": self.ngram_range,
            "stop_words": self.stop_words,
            "lowercase": self.lowercase,
        }

        if self.vectorizer_type == "tfidf":
            base_params.update(
                {
                    "use_idf": self.use_idf,
                    "smooth_idf": self.smooth_idf,
                    "sublinear_tf": self.sublinear_tf,
                    "norm": self.norm,
                }
            )
        elif self.vectorizer_type == "count":
            base_params["binary"] = self.binary

        return base_params


@dataclass
class UMAPConfig:
    """Configuration for UMAP dimensionality reduction."""

    n_neighbors: int = 15
    n_components: int = 5
    min_dist: float = 0.1
    metric: str = "cosine"
    spread: float = 1.0
    low_memory: bool = False
    random_state: Optional[int] = 42
    verbose: bool = False

    # For visualization
    n_components_viz: int = 2  # For 2D visualization
    n_components_3d: int = 3  # For 3D visualization

    def get_umap_params(
        self, for_visualization: bool = False, use_3d: bool = False
    ) -> Dict[str, Any]:
        """Get UMAP parameters."""
        n_comp = self.n_components
        if for_visualization:
            n_comp = self.n_components_3d if use_3d else self.n_components_viz

        return {
            "n_neighbors": self.n_neighbors,
            "n_components": n_comp,
            "min_dist": self.min_dist,
            "metric": self.metric,
            "spread": self.spread,
            "low_memory": self.low_memory,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }


@dataclass
class RepresentationConfig:
    """Configuration for representation models."""

    use_representation: bool = True
    representation_models: List[str] = field(default_factory=lambda: ["KeyBERT"])

    # KeyBERT parameters
    keybert_model: Optional[str] = None  # Use embedding model if None
    keybert_top_k: int = 10
    keybert_use_mmr: bool = False
    keybert_diversity: float = 0.3

    # MaximalMarginalRelevance parameters
    mmr_diversity: float = 0.1
    mmr_top_k: int = 10

    # Part-of-Speech parameters
    pos_patterns: str = "<J.*>*<N.*>+"  # Adjectives followed by nouns
    pos_top_k: int = 10

    # c-TF-IDF parameters (always used)
    ctfidf_reduce_frequent_words: bool = False
    ctfidf_bm25_weighting: bool = False

    def get_active_models(self) -> List[str]:
        """Get list of active representation models."""
        if not self.use_representation:
            return []
        return self.representation_models.copy()


@dataclass
class TopicModelConfig:
    """Configuration for BERTopic model training."""

    # Core configuration
    embedding_config: Optional[EmbeddingConfig] = None
    clustering_config: ClusteringConfig = field(default_factory=ClusteringConfig)
    vectorization_config: VectorizationConfig = field(
        default_factory=VectorizationConfig
    )
    umap_config: UMAPConfig = field(default_factory=UMAPConfig)
    representation_config: RepresentationConfig = field(
        default_factory=RepresentationConfig
    )

    # Advanced options
    top_k_words: int = 10
    nr_topics: Optional[int] = None  # Auto-reduce to this many topics
    low_memory: bool = False
    calculate_probabilities: bool = False
    verbose: bool = True

    # Guided topic modeling
    use_guided_modeling: bool = False
    seed_topic_list: List[List[str]] = field(default_factory=list)

    # Model persistence
    model_name: str = "bertopic_model"
    save_model: bool = True

    @property
    def is_configured(self) -> bool:
        """Check if configuration is valid for training."""
        import logging
        logger = logging.getLogger(__name__)
        
        if self.embedding_config is None:
            logger.debug("TopicModelConfig.is_configured: embedding_config is None")
            return False
        
        embedding_configured = self.embedding_config.is_configured
        logger.debug(f"TopicModelConfig.is_configured: embedding_config.is_configured = {embedding_configured}")
        
        return embedding_configured

    @property
    def estimated_memory_gb(self) -> float:
        """Estimate memory requirements in GB."""
        if not self.embedding_config or not self.embedding_config.model_info:
            return 1.0

        # Base model memory
        model_memory = self.embedding_config.model_info.memory_usage_mb or 500

        # Embedding memory (rough estimate)
        embedding_memory = 0.1  # Default if no info
        if self.embedding_config.model_info.embedding_dimension:
            # Estimate for 10K documents
            embedding_memory = (
                10000 * self.embedding_config.model_info.embedding_dimension * 4
            ) / (1024 * 1024)

        # UMAP and clustering overhead
        processing_overhead = 200

        total_mb = model_memory + embedding_memory + processing_overhead
        return total_mb / 1024  # Convert to GB


@dataclass
class TopicInfo:
    """Information about a single topic."""

    topic_id: int
    size: int  # Number of documents
    words: List[Tuple[str, float]]  # (word, importance_score)
    representative_docs: List[str] = field(default_factory=list)
    name: Optional[str] = None

    @property
    def top_words(self) -> List[str]:
        """Get list of top words for this topic."""
        return [word for word, _ in self.words]

    @property
    def top_words_string(self) -> str:
        """Get top words as a formatted string."""
        return ", ".join(self.top_words[:5])

    @property
    def percentage(self) -> float:
        """Percentage of documents in this topic (set by TopicResult)."""
        return getattr(self, "_percentage", 0.0)


@dataclass
class TopicResult:
    """Results from topic modeling."""

    # Core results
    topics: List[int]  # Topic assignment for each document
    probabilities: Optional[List[List[float]]] = (
        None  # Topic probabilities per document
    )
    topic_info: List[TopicInfo] = field(default_factory=list)

    # Model artifacts
    embeddings: Optional[np.ndarray] = None
    umap_embeddings: Optional[np.ndarray] = None  # 2D/3D embeddings for visualization

    # Training metadata
    config: Optional[TopicModelConfig] = None
    training_time_seconds: float = 0.0
    model_path: Optional[str] = None

    # Quality metrics
    silhouette_score: Optional[float] = None
    calinski_harabasz_score: Optional[float] = None
    davies_bouldin_score: Optional[float] = None

    @property
    def num_documents(self) -> int:
        """Number of documents processed."""
        return len(self.topics)

    @property
    def num_topics(self) -> int:
        """Number of topics found (excluding outliers)."""
        return len([t for t in self.topic_info if t.topic_id != -1])

    @property
    def outlier_count(self) -> int:
        """Number of outlier documents (topic -1)."""
        return sum(1 for t in self.topics if t == -1)

    @property
    def outlier_percentage(self) -> float:
        """Percentage of documents classified as outliers."""
        if self.num_documents == 0:
            return 0.0
        return (self.outlier_count / self.num_documents) * 100

    def update_topic_percentages(self):
        """Update percentage information for each topic."""
        if not self.topic_info or self.num_documents == 0:
            return

        for topic in self.topic_info:
            topic._percentage = (topic.size / self.num_documents) * 100

    def get_topic_by_id(self, topic_id: int) -> Optional[TopicInfo]:
        """Get topic information by ID."""
        for topic in self.topic_info:
            if topic.topic_id == topic_id:
                return topic
        return None


@dataclass
class TopicModelingProgress:
    """Progress tracking for topic modeling operations."""

    stage: str = "initializing"  # "initializing", "embedding", "clustering", "representation", "complete", "error"
    progress_percentage: float = 0.0
    current_step: str = ""
    total_steps: int = 5
    current_step_num: int = 0
    elapsed_time_seconds: float = 0.0
    estimated_remaining_seconds: Optional[float] = None
    error_message: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        """Check if processing is complete."""
        return self.stage == "complete"

    @property
    def has_error(self) -> bool:
        """Check if there was an error."""
        return self.stage == "error"

    @property
    def progress_text(self) -> str:
        """Get formatted progress text."""
        if self.has_error:
            return f"Error: {self.error_message}"
        elif self.is_complete:
            return "Topic modeling complete!"
        else:
            return (
                f"Step {self.current_step_num}/{self.total_steps}: {self.current_step}"
            )
