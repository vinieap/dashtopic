"""
BERTopic Service - Core topic modeling functionality.

This service provides the main interface for training and managing BERTopic models.
It handles async training, progress reporting, model configuration, and result generation.

Features:
- Asynchronous model training with progress callbacks
- Support for custom embeddings and embedding models
- Comprehensive error handling and cancellation
- Model persistence and loading
- Quality metrics calculation
- Memory management and cleanup

Classes:
    BERTopicService: Main service class for BERTopic operations

Example:
    >>> service = BERTopicService()
    >>> service.set_callbacks(progress_callback=update_progress)
    >>> service.train_model_async(texts, config, embeddings)
"""

import logging
import time
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
import threading
import gc

# BERTopic and ML imports
try:
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import (
        silhouette_score,
        calinski_harabasz_score,
        davies_bouldin_score,
    )
    from umap import UMAP
    import hdbscan
    from bertopic.representation import (
        KeyBERTInspired,
        MaximalMarginalRelevance,
        PartOfSpeech,
    )
except ImportError as e:
    logging.error(f"Required ML libraries not installed: {e}")
    raise

from ..models import (
    TopicModelConfig,
    TopicResult,
    TopicInfo,
    TopicModelingProgress,
    ClusteringConfig,
    VectorizationConfig,
    UMAPConfig,
    RepresentationConfig,
)

logger = logging.getLogger(__name__)


class BERTopicService:
    """
    Service for BERTopic model training and management.
    
    This service provides a high-level interface for training BERTopic models with
    comprehensive configuration options, progress tracking, and error handling.
    
    Attributes:
        current_model: The currently loaded BERTopic model instance
        is_training: Flag indicating if training is in progress
        progress_callback: Callback for training progress updates
        completion_callback: Callback for training completion
        error_callback: Callback for error handling
        
    Thread Safety:
        This service uses background threads for training. Multiple training
        operations cannot run simultaneously - attempting to start training
        while another is in progress will result in an error callback.
    """

    def __init__(self) -> None:
        """
        Initialize the BERTopic service.
        
        Sets up internal state for model management and training coordination.
        """
        self.current_model: Optional[BERTopic] = None
        self.is_training: bool = False
        self._training_thread: Optional[threading.Thread] = None
        self._cancel_training: bool = False

        # Callbacks for UI integration
        self.progress_callback: Optional[Callable[[TopicModelingProgress], None]] = None
        self.completion_callback: Optional[Callable[[TopicResult], None]] = None
        self.error_callback: Optional[Callable[[str], None]] = None

        logger.info("BERTopic service initialized")

    def set_callbacks(
        self,
        progress_callback: Optional[Callable[[TopicModelingProgress], None]] = None,
        completion_callback: Optional[Callable[[TopicResult], None]] = None,
        error_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Set callback functions for training events.
        
        Args:
            progress_callback: Called with TopicModelingProgress during training
            completion_callback: Called with TopicResult when training completes
            error_callback: Called with error message string if training fails
            
        Note:
            Callbacks are called from the background training thread.
        """
        self.progress_callback = progress_callback
        self.completion_callback = completion_callback
        self.error_callback = error_callback

    def train_model_async(
        self,
        texts: List[str],
        config: TopicModelConfig,
        embeddings: Optional[np.ndarray] = None,
        embedding_model: Optional[Any] = None,
    ) -> None:
        """
        Start model training in a background thread.
        
        Args:
            texts: List of documents to analyze
            config: Topic modeling configuration with all parameters
            embeddings: Pre-computed embeddings (optional, will compute if None)
            embedding_model: Custom embedding model (optional)
            
        Raises:
            ValueError: If training is already in progress (via error callback)
            
        Note:
            This method returns immediately. Use callbacks to track progress
            and completion. Only one training operation can run at a time.
        """
        if self.is_training:
            if self.error_callback:
                self.error_callback("Training already in progress")
            return

        self.is_training = True
        self._cancel_training = False

        self._training_thread = threading.Thread(
            target=self._train_model_worker,
            args=(texts, config, embeddings, embedding_model),
            daemon=True,
        )
        self._training_thread.start()
        logger.info("Started background topic modeling training")

    def cancel_training(self) -> None:
        """
        Cancel ongoing training operation.
        
        Sets a cancellation flag that is checked during training stages.
        The training thread will exit gracefully at the next checkpoint.
        
        Note:
            Cancellation is not immediate - it may take time for the training
            thread to check the flag and exit cleanly.
        """
        if self.is_training:
            self._cancel_training = True
            logger.info("Training cancellation requested")

    def _train_model_worker(
        self,
        texts: List[str],
        config: TopicModelConfig,
        embeddings: Optional[np.ndarray] = None,
        embedding_model=None,
    ) -> None:
        """Worker method for background training."""
        try:
            result = self.train_model(texts, config, embeddings, embedding_model)
            if result and self.completion_callback and not self._cancel_training:
                self.completion_callback(result)
        except Exception as e:
            logger.error(f"Training worker error: {e}", exc_info=True)
            if self.error_callback:
                self.error_callback(str(e))
        finally:
            self.is_training = False

    def train_model(
        self,
        texts: List[str],
        config: TopicModelConfig,
        embeddings: Optional[np.ndarray] = None,
        embedding_model=None,
    ) -> Optional[TopicResult]:
        """Train a BERTopic model with the given configuration."""
        logger.debug("BERTopic train_model: Starting configuration validation")
        logger.debug(f"BERTopic train_model: config.is_configured = {config.is_configured}")
        
        if not config.is_configured:
            logger.error("BERTopic train_model: Configuration validation failed")
            logger.error(f"BERTopic train_model: embedding_config exists = {config.embedding_config is not None}")
            if config.embedding_config:
                logger.error(f"BERTopic train_model: model_info exists = {config.embedding_config.model_info is not None}")
                if config.embedding_config.model_info:
                    logger.error(f"BERTopic train_model: model_info.is_loaded = {config.embedding_config.model_info.is_loaded}")
                    logger.error(f"BERTopic train_model: model_info.model_type = {config.embedding_config.model_info.model_type}")
            raise ValueError("Invalid topic modeling configuration")

        start_time = time.time()
        progress = TopicModelingProgress(
            stage="initializing",
            current_step="Preparing model components",
            current_step_num=1,
            total_steps=5,
        )

        try:
            # Step 1: Initialize components
            self._update_progress(progress, 0, "Initializing model components")
            if self._cancel_training:
                return None

            # Create clustering model
            clustering_model = self._create_clustering_model(config.clustering_config)

            # Create vectorization model
            vectorizer_model = self._create_vectorizer_model(
                config.vectorization_config
            )

            # Create UMAP model
            umap_model = self._create_umap_model(config.umap_config)

            # Create representation models
            representation_models = self._create_representation_models(
                config.representation_config
            )

            # Step 2: Prepare embeddings
            self._update_progress(progress, 20, "Preparing embeddings", 2)
            if self._cancel_training:
                return None

            if embeddings is None:
                if self.error_callback:
                    self.error_callback("No embeddings provided for training")
                return None

            # Step 3: Initialize BERTopic model
            self._update_progress(progress, 40, "Creating BERTopic model", 3)
            if self._cancel_training:
                return None

            # Handle embedding model for representation models
            final_embedding_model = embedding_model

            # If no embedding model provided but representation models are configured
            # Skip loading for precomputed embeddings during optimization
            if (
                embedding_model is None
                and config.representation_config.use_representation
                and config.embedding_config
                and config.embedding_config.model_info
                and config.embedding_config.model_info.model_type != "precomputed"
                and config.embedding_config.model_info.is_loaded
            ):
                logger.info("Loading embedding model for representation models")
                try:
                    from .model_management_service import ModelManagementService

                    model_service = ModelManagementService()

                    if model_service.load_model(config.embedding_config.model_info):
                        final_embedding_model = model_service.get_loaded_model(
                            config.embedding_config.model_info.model_name
                        )
                        logger.info(
                            "Successfully loaded embedding model for representation"
                        )
                    else:
                        logger.warning(
                            "Failed to load embedding model - representation models may not work"
                        )

                except Exception as e:
                    logger.warning(
                        f"Could not load embedding model for representation: {e}"
                    )

            topic_model = BERTopic(
                embedding_model=final_embedding_model,  # Provide actual embedding model for representation models
                umap_model=umap_model,
                hdbscan_model=clustering_model,
                vectorizer_model=vectorizer_model,
                representation_model=representation_models,
                nr_topics=config.nr_topics,
                low_memory=config.low_memory,
                calculate_probabilities=config.calculate_probabilities,
                verbose=config.verbose,
            )

            # Step 4: Train the model
            self._update_progress(progress, 60, "Training topic model", 4)
            if self._cancel_training:
                return None

            # Handle guided topic modeling
            guided_topics = None
            if config.use_guided_modeling and config.seed_topic_list:
                guided_topics = config.seed_topic_list

            # Fit the model
            if guided_topics:
                topics, probabilities = topic_model.fit_transform(
                    texts, embeddings, y=guided_topics
                )
            else:
                topics, probabilities = topic_model.fit_transform(texts, embeddings)

            # Step 5: Process results
            self._update_progress(progress, 80, "Processing results", 5)
            if self._cancel_training:
                return None

            # Create visualization embeddings
            viz_umap_model = self._create_umap_model(
                config.umap_config, for_visualization=True
            )
            umap_embeddings = viz_umap_model.fit_transform(embeddings)

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(embeddings, topics)

            # Extract topic information
            topic_info = self._extract_topic_info(topic_model, topics)

            # Create result object
            result = TopicResult(
                topics=topics.tolist() if hasattr(topics, "tolist") else topics,
                probabilities=probabilities.tolist()
                if probabilities is not None
                else None,
                topic_info=topic_info,
                embeddings=embeddings,
                umap_embeddings=umap_embeddings,
                config=config,
                training_time_seconds=time.time() - start_time,
                silhouette_score=quality_metrics.get("silhouette"),
                calinski_harabasz_score=quality_metrics.get("calinski_harabasz"),
                davies_bouldin_score=quality_metrics.get("davies_bouldin"),
            )

            # Update topic percentages
            result.update_topic_percentages()

            # Save model if requested
            if config.save_model:
                model_path = self._save_model(topic_model, config)
                result.model_path = model_path

            # Store current model
            self.current_model = topic_model

            # Final progress update
            self._update_progress(progress, 100, "Training complete", 5)
            progress.stage = "complete"
            if self.progress_callback:
                self.progress_callback(progress)

            logger.info(
                f"Topic modeling completed in {result.training_time_seconds:.2f}s"
            )
            logger.info(
                f"Found {result.num_topics} topics with {result.outlier_percentage:.1f}% outliers"
            )

            return result

        except Exception as e:
            progress.stage = "error"
            progress.error_message = str(e)
            if self.progress_callback:
                self.progress_callback(progress)
            logger.error(f"Topic modeling failed: {e}", exc_info=True)
            raise
        finally:
            # Cleanup
            gc.collect()

    def _update_progress(
        self,
        progress: TopicModelingProgress,
        percentage: float,
        step: str,
        step_num: Optional[int] = None,
    ):
        """Update and broadcast progress."""
        progress.progress_percentage = percentage
        progress.current_step = step
        if step_num is not None:
            progress.current_step_num = step_num
        progress.elapsed_time_seconds = time.time() - (
            progress.elapsed_time_seconds or time.time()
        )

        if self.progress_callback:
            self.progress_callback(progress)

    def _create_clustering_model(self, config: ClusteringConfig):
        """Create clustering model based on configuration."""
        params = config.get_algorithm_params()

        if config.algorithm == "hdbscan":
            return hdbscan.HDBSCAN(**params)
        elif config.algorithm == "kmeans":
            return KMeans(**params)
        elif config.algorithm == "agglomerative":
            return AgglomerativeClustering(**params)
        elif config.algorithm == "optics":
            try:
                from sklearn.cluster import OPTICS

                return OPTICS(**params)
            except ImportError:
                logger.warning("OPTICS not available, falling back to HDBSCAN")
                return hdbscan.HDBSCAN(min_cluster_size=config.min_cluster_size)
        else:
            logger.warning(
                f"Unknown clustering algorithm: {config.algorithm}, using HDBSCAN"
            )
            return hdbscan.HDBSCAN(min_cluster_size=config.min_cluster_size)

    def _create_vectorizer_model(self, config: VectorizationConfig):
        """Create vectorizer model based on configuration."""
        params = config.get_vectorizer_params()

        if config.vectorizer_type == "tfidf":
            return TfidfVectorizer(**params)
        elif config.vectorizer_type == "count":
            return CountVectorizer(**params)
        else:
            logger.warning(
                f"Unknown vectorizer: {config.vectorizer_type}, using TF-IDF"
            )
            return TfidfVectorizer(**params)

    def _create_umap_model(self, config: UMAPConfig, for_visualization: bool = False):
        """Create UMAP model based on configuration."""
        params = config.get_umap_params(for_visualization=for_visualization)
        return UMAP(**params)

    def _create_representation_models(
        self, config: RepresentationConfig
    ) -> Dict[str, Any]:
        """Create representation models based on configuration."""
        if not config.use_representation:
            return {}

        models = {}
        active_models = config.get_active_models()

        for model_name in active_models:
            try:
                if model_name == "KeyBERT":
                    models["KeyBERT"] = KeyBERTInspired()
                elif model_name == "MaximalMarginalRelevance":
                    models["MMR"] = MaximalMarginalRelevance(
                        diversity=config.mmr_diversity
                    )
                elif model_name == "PartOfSpeech":
                    try:
                        models["POS"] = PartOfSpeech(model="en_core_web_sm")
                    except OSError:
                        logger.warning(
                            "spaCy model not found, skipping PartOfSpeech representation"
                        )

            except Exception as e:
                logger.warning(
                    f"Failed to create {model_name} representation model: {e}"
                )

        return models

    def _calculate_quality_metrics(
        self, embeddings: np.ndarray, topics: List[int]
    ) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        metrics = {}

        try:
            # Filter out outliers (topic -1) for metrics calculation
            mask = np.array(topics) != -1
            if np.sum(mask) < 2:
                logger.warning("Not enough non-outlier points for quality metrics")
                return metrics

            filtered_embeddings = embeddings[mask]
            filtered_topics = np.array(topics)[mask]

            # Only calculate if we have multiple clusters
            unique_topics = np.unique(filtered_topics)
            if len(unique_topics) > 1:
                # Silhouette score
                metrics["silhouette"] = silhouette_score(
                    filtered_embeddings, filtered_topics
                )

                # Calinski-Harabasz score
                metrics["calinski_harabasz"] = calinski_harabasz_score(
                    filtered_embeddings, filtered_topics
                )

                # Davies-Bouldin score (lower is better)
                metrics["davies_bouldin"] = davies_bouldin_score(
                    filtered_embeddings, filtered_topics
                )

        except Exception as e:
            logger.warning(f"Failed to calculate quality metrics: {e}")

        return metrics

    def _extract_topic_info(
        self, model: BERTopic, topics: List[int]
    ) -> List[TopicInfo]:
        """Extract topic information from trained model."""
        topic_info_list = []

        try:
            # Get topic information from BERTopic
            topic_info_df = model.get_topic_info()

            for _, row in topic_info_df.iterrows():
                topic_id = int(row["Topic"])

                # Get topic words
                topic_words = model.get_topic(topic_id)
                if not topic_words:
                    continue

                # Get representative documents
                try:
                    representative_docs = model.get_representative_docs(topic_id)
                    if representative_docs and len(representative_docs) > 0:
                        rep_docs = representative_docs[:3]  # Top 3 representative docs
                    else:
                        rep_docs = []
                except:
                    rep_docs = []

                topic_info = TopicInfo(
                    topic_id=topic_id,
                    size=int(row["Count"]),
                    words=topic_words,
                    representative_docs=rep_docs,
                    name=row.get("Name", f"Topic {topic_id}"),
                )

                topic_info_list.append(topic_info)

        except Exception as e:
            logger.error(f"Failed to extract topic info: {e}")

        return topic_info_list

    def _save_model(self, model: BERTopic, config: TopicModelConfig) -> str:
        """Save trained model to disk."""
        try:
            # Create models directory if it doesn't exist
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)

            # Generate unique filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_filename = f"{config.model_name}_{timestamp}.pkl"
            model_path = models_dir / model_filename

            # Save model
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            logger.info(f"Model saved to {model_path}")
            return str(model_path)

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return ""

    def load_model(self, model_path: str) -> bool:
        """Load a saved BERTopic model."""
        try:
            with open(model_path, "rb") as f:
                self.current_model = pickle.load(f)

            logger.info(f"Model loaded from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_topic_words(
        self, topic_id: int, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Get top words for a specific topic."""
        if not self.current_model:
            return []

        try:
            return self.current_model.get_topic(topic_id)[:top_k]
        except Exception as e:
            logger.error(f"Failed to get topic words: {e}")
            return []

    def get_document_topics(
        self, texts: List[str], embeddings: np.ndarray
    ) -> List[int]:
        """Get topic assignments for new documents."""
        if not self.current_model:
            return []

        try:
            topics, _ = self.current_model.transform(texts, embeddings)
            return topics.tolist()
        except Exception as e:
            logger.error(f"Failed to get document topics: {e}")
            return []

    def cleanup(self):
        """Cleanup resources."""
        if self.is_training:
            self.cancel_training()

        self.current_model = None
        gc.collect()
        logger.info("BERTopic service cleaned up")
