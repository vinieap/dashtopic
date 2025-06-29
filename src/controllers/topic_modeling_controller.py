"""
Topic Modeling Controller - Coordinates topic modeling workflow.
"""

import logging
from typing import Optional, Callable, List, Dict, Any
import gc

from ..services import BERTopicService, EmbeddingService, DataValidationService
from ..models import (
    TopicModelConfig,
    TopicResult,
    TopicModelingProgress,
    EmbeddingResult,
    DataConfig,
    ClusteringConfig,
    VectorizationConfig,
    UMAPConfig,
    RepresentationConfig,
)

logger = logging.getLogger(__name__)


class TopicModelingController:
    """Controller for managing topic modeling operations."""

    def __init__(self):
        # Services
        self.bertopic_service = BERTopicService()
        self.embedding_service: Optional[EmbeddingService] = None
        self.data_validation_service: Optional[DataValidationService] = None

        # State
        self.current_config: Optional[TopicModelConfig] = None
        self.current_result: Optional[TopicResult] = None
        self.is_processing = False

        # UI Callbacks
        self.progress_callback: Optional[Callable[[TopicModelingProgress], None]] = None
        self.completion_callback: Optional[Callable[[TopicResult], None]] = None
        self.error_callback: Optional[Callable[[str], None]] = None
        self.status_callback: Optional[Callable[[str], None]] = None

        # Setup service callbacks
        self._setup_service_callbacks()

        logger.info("Topic modeling controller initialized")

    def set_services(
        self,
        embedding_service: EmbeddingService,
        data_validation_service: DataValidationService,
    ):
        """Set required services."""
        self.embedding_service = embedding_service
        self.data_validation_service = data_validation_service
        logger.info("Services configured for topic modeling controller")

    def set_callbacks(
        self,
        progress_callback: Optional[Callable[[TopicModelingProgress], None]] = None,
        completion_callback: Optional[Callable[[TopicResult], None]] = None,
        error_callback: Optional[Callable[[str], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ):
        """Set UI callback functions."""
        self.progress_callback = progress_callback
        self.completion_callback = completion_callback
        self.error_callback = error_callback
        self.status_callback = status_callback

        # Update service callbacks
        self._setup_service_callbacks()

    def _setup_service_callbacks(self):
        """Setup callbacks for BERTopic service."""
        self.bertopic_service.set_callbacks(
            progress_callback=self._on_training_progress,
            completion_callback=self._on_training_complete,
            error_callback=self._on_training_error,
        )

    def create_default_config(self) -> TopicModelConfig:
        """Create a default topic modeling configuration."""
        return TopicModelConfig(
            clustering_config=ClusteringConfig(),
            vectorization_config=VectorizationConfig(),
            umap_config=UMAPConfig(),
            representation_config=RepresentationConfig(),
            top_k_words=10,
            verbose=True,
            save_model=True,
        )

    def validate_configuration(
        self, config: TopicModelConfig
    ) -> tuple[bool, List[str]]:
        """Validate topic modeling configuration."""
        errors = []

        # Check if embedding configuration is valid
        if not config.embedding_config or not config.embedding_config.is_configured:
            errors.append("No embedding model configured or model not loaded")

        # Validate clustering configuration
        try:
            clustering_params = config.clustering_config.get_algorithm_params()
            if not clustering_params:
                errors.append(
                    f"Invalid clustering algorithm: {config.clustering_config.algorithm}"
                )
        except Exception as e:
            errors.append(f"Clustering configuration error: {str(e)}")

        # Validate vectorization configuration
        try:
            vectorizer_params = config.vectorization_config.get_vectorizer_params()
            if not vectorizer_params:
                errors.append("Invalid vectorization configuration")
        except Exception as e:
            errors.append(f"Vectorization configuration error: {str(e)}")

        # Validate UMAP configuration
        try:
            umap_params = config.umap_config.get_umap_params()
            if config.umap_config.n_neighbors < 2:
                errors.append("UMAP n_neighbors must be at least 2")
            if config.umap_config.n_components < 2:
                errors.append("UMAP n_components must be at least 2")
        except Exception as e:
            errors.append(f"UMAP configuration error: {str(e)}")

        # Check memory requirements
        try:
            estimated_memory = config.estimated_memory_gb
            if estimated_memory > 8.0:  # Warning threshold
                errors.append(f"High memory usage estimated: {estimated_memory:.1f}GB")
        except Exception as e:
            logger.warning(f"Could not estimate memory usage: {e}")

        is_valid = len(errors) == 0
        return is_valid, errors

    def start_topic_modeling(
        self,
        data_config: DataConfig,
        topic_config: TopicModelConfig,
        embedding_result: EmbeddingResult,
    ) -> bool:
        """Start topic modeling process."""
        if self.is_processing:
            self._notify_error("Topic modeling already in progress")
            return False

        # Validate configuration
        is_valid, errors = self.validate_configuration(topic_config)
        if not is_valid:
            error_msg = "Configuration validation failed:\n" + "\n".join(errors)
            self._notify_error(error_msg)
            return False

        # Validate data and embeddings
        if not self._validate_data_and_embeddings(data_config, embedding_result):
            return False

        # Store configuration
        self.current_config = topic_config
        self.is_processing = True

        try:
            # Prepare texts
            texts = self._prepare_texts(data_config, embedding_result)
            if not texts:
                self._notify_error("Failed to prepare text data")
                return False

            self._notify_status("Starting topic modeling...")

            # Get the embedding model for representation models
            embedding_model = None

            # Try to get embedding model from multiple sources
            if self.embedding_service and hasattr(
                self.embedding_service, "current_model"
            ):
                embedding_model = self.embedding_service.current_model

            # If not found, try to get from main window's embedding service
            if embedding_model is None:
                try:
                    # Get main window reference through widget hierarchy
                    main_window = self._get_main_window()
                    if main_window and hasattr(main_window, "embedding_service"):
                        main_embedding_service = main_window.embedding_service
                        if hasattr(main_embedding_service, "current_model"):
                            embedding_model = main_embedding_service.current_model
                except Exception as e:
                    logger.warning(
                        f"Could not access main window embedding service: {e}"
                    )

            # If still not found, try to load the model directly from embedding result
            if embedding_model is None and embedding_result.model_info:
                try:
                    from ..services.model_management_service import (
                        ModelManagementService,
                    )

                    temp_model_service = ModelManagementService()
                    if temp_model_service.load_model(embedding_result.model_info):
                        embedding_model = temp_model_service.get_loaded_model(
                            embedding_result.model_info.model_name
                        )
                except Exception as e:
                    logger.warning(f"Could not load embedding model directly: {e}")

            logger.info(
                f"Embedding model for representation: {type(embedding_model)} - {embedding_model is not None}"
            )
            if embedding_model is None:
                logger.warning(
                    "No embedding model available for representation models - they may not work properly"
                )

            # Add embedding config to topic config for representation models
            if not hasattr(topic_config, "embedding_config"):
                topic_config.embedding_config = embedding_result.config

            # Start training asynchronously
            self.bertopic_service.train_model_async(
                texts=texts,
                config=topic_config,
                embeddings=embedding_result.embeddings,
                embedding_model=embedding_model,
            )

            logger.info("Topic modeling started successfully")
            return True

        except Exception as e:
            self.is_processing = False
            error_msg = f"Failed to start topic modeling: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._notify_error(error_msg)
            return False

    def cancel_topic_modeling(self) -> None:
        """Cancel ongoing topic modeling."""
        if self.is_processing:
            self.bertopic_service.cancel_training()
            self._notify_status("Cancelling topic modeling...")
            logger.info("Topic modeling cancellation requested")

    def _validate_data_and_embeddings(
        self, data_config: DataConfig, embedding_result: EmbeddingResult
    ) -> bool:
        """Validate that data and embeddings are compatible."""
        try:
            # Check data configuration
            if not data_config.is_configured:
                self._notify_error("Data is not properly configured")
                return False

            # Check embedding result
            if not embedding_result or embedding_result.embeddings is None:
                self._notify_error("No embeddings available for topic modeling")
                return False

            # Check if text count matches embedding count
            expected_count = len(embedding_result.texts)
            actual_embedding_count = embedding_result.embeddings.shape[0]

            if expected_count != actual_embedding_count:
                self._notify_error(
                    f"Text count ({expected_count}) doesn't match embedding count ({actual_embedding_count})"
                )
                return False

            # Check minimum requirements
            if expected_count < 10:
                self._notify_error(
                    "At least 10 documents are required for topic modeling"
                )
                return False

            logger.info(
                f"Data validation passed: {expected_count} documents ready for topic modeling"
            )
            return True

        except Exception as e:
            error_msg = f"Data validation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._notify_error(error_msg)
            return False

    def _prepare_texts(
        self, data_config: DataConfig, embedding_result: EmbeddingResult
    ) -> List[str]:
        """Prepare texts for topic modeling."""
        try:
            # Use texts from embedding result (they should already be prepared)
            texts = embedding_result.texts

            # Additional text preparation if needed
            prepared_texts = []
            for text in texts:
                # Basic cleaning
                cleaned_text = str(text).strip()
                if len(cleaned_text) >= data_config.min_text_length:
                    # Apply max length if specified
                    if data_config.max_text_length:
                        cleaned_text = cleaned_text[: data_config.max_text_length]
                    prepared_texts.append(cleaned_text)

            if len(prepared_texts) < len(texts):
                logger.warning(
                    f"Filtered out {len(texts) - len(prepared_texts)} texts due to length constraints"
                )

            return prepared_texts

        except Exception as e:
            logger.error(f"Failed to prepare texts: {e}", exc_info=True)
            return []

    def _on_training_progress(self, progress: TopicModelingProgress):
        """Handle training progress updates."""
        if self.progress_callback:
            self.progress_callback(progress)

        # Update status
        status_msg = f"Topic Modeling: {progress.progress_text}"
        self._notify_status(status_msg)

    def _on_training_complete(self, result: TopicResult):
        """Handle training completion."""
        self.is_processing = False
        self.current_result = result

        # Log results
        logger.info("Topic modeling completed successfully:")
        logger.info(f"- Found {result.num_topics} topics")
        logger.info(f"- {result.outlier_percentage:.1f}% outliers")
        logger.info(f"- Training time: {result.training_time_seconds:.2f}s")

        if result.silhouette_score:
            logger.info(f"- Silhouette score: {result.silhouette_score:.3f}")

        # Notify UI
        self._notify_status(
            f"Topic modeling complete! Found {result.num_topics} topics"
        )

        if self.completion_callback:
            self.completion_callback(result)

    def _on_training_error(self, error_message: str):
        """Handle training errors."""
        self.is_processing = False
        logger.error(f"Topic modeling failed: {error_message}")
        self._notify_error(f"Topic modeling failed: {error_message}")

    def _notify_status(self, message: str):
        """Notify status update."""
        if self.status_callback:
            self.status_callback(message)

    def _notify_error(self, message: str):
        """Notify error."""
        if self.error_callback:
            self.error_callback(message)

    def get_current_result(self) -> Optional[TopicResult]:
        """Get the current topic modeling result."""
        return self.current_result

    def get_current_config(self) -> Optional[TopicModelConfig]:
        """Get the current topic modeling configuration."""
        return self.current_config

    def load_saved_model(self, model_path: str) -> bool:
        """Load a previously saved model."""
        try:
            success = self.bertopic_service.load_model(model_path)
            if success:
                self._notify_status(f"Model loaded from {model_path}")
                logger.info(f"Successfully loaded model from {model_path}")
            else:
                self._notify_error(f"Failed to load model from {model_path}")
            return success
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._notify_error(error_msg)
            return False

    def get_topic_words(
        self, topic_id: int, top_k: int = 10
    ) -> List[tuple[str, float]]:
        """Get top words for a specific topic."""
        try:
            return self.bertopic_service.get_topic_words(topic_id, top_k)
        except Exception as e:
            logger.error(f"Failed to get topic words: {e}")
            return []

    def predict_topics(self, texts: List[str], embeddings) -> List[int]:
        """Predict topics for new documents."""
        try:
            return self.bertopic_service.get_document_topics(texts, embeddings)
        except Exception as e:
            logger.error(f"Failed to predict topics: {e}")
            return []

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            "is_processing": self.is_processing,
            "has_model": self.bertopic_service.current_model is not None,
            "has_result": self.current_result is not None,
            "has_config": self.current_config is not None,
        }

    def clear_current_results(self):
        """Clear current results and model."""
        self.current_result = None
        self.current_config = None
        self.bertopic_service.current_model = None
        self._notify_status("Results cleared")
        logger.info("Topic modeling results cleared")

    def update_config_from_parameters(self, parameters: Dict[str, Any]):
        """Update the current configuration from optimization parameters.
        
        Args:
            parameters: Dictionary of parameter names to values
        """
        if not self.current_config:
            logger.warning("No current config to update")
            return
        
        # Update clustering parameters
        if "min_cluster_size" in parameters:
            self.current_config.clustering.min_cluster_size = parameters["min_cluster_size"]
        if "min_samples" in parameters:
            self.current_config.clustering.min_samples = parameters["min_samples"]
        if "metric" in parameters:
            self.current_config.clustering.metric = parameters["metric"]
        if "n_clusters" in parameters:
            self.current_config.clustering.n_clusters = parameters["n_clusters"]
        
        # Update UMAP parameters
        if "n_neighbors" in parameters:
            self.current_config.umap.n_neighbors = parameters["n_neighbors"]
        if "n_components" in parameters:
            self.current_config.umap.n_components = parameters["n_components"]
        if "min_dist" in parameters:
            self.current_config.umap.min_dist = parameters["min_dist"]
        
        # Update vectorization parameters
        if "min_df" in parameters:
            self.current_config.vectorization.min_df = parameters["min_df"]
        if "max_df" in parameters:
            self.current_config.vectorization.max_df = parameters["max_df"]
        if "ngram_range" in parameters:
            self.current_config.vectorization.ngram_range = parameters["ngram_range"]
        
        # Update BERTopic parameters
        if "top_n_words" in parameters:
            self.current_config.top_n_words = parameters["top_n_words"]
        if "nr_topics" in parameters:
            self.current_config.nr_topics = parameters["nr_topics"]
        
        logger.info("Updated topic model config from optimization parameters")

    def _get_main_window(self):
        """Get reference to main window (helper method)."""
        # This is a simple approach - in practice you might want to store the reference
        # during initialization or use a different pattern
        try:
            import tkinter as tk

            for widget in tk._default_root.winfo_children():
                if hasattr(widget, "get_data_config"):  # Main window identifier
                    return widget
        except:
            pass
        return None

    def cleanup(self):
        """Cleanup resources."""
        if self.is_processing:
            self.cancel_topic_modeling()

        self.bertopic_service.cleanup()
        self.current_result = None
        self.current_config = None

        # Clear callbacks
        self.progress_callback = None
        self.completion_callback = None
        self.error_callback = None
        self.status_callback = None

        gc.collect()
        logger.info("Topic modeling controller cleaned up")
