"""
Embedding Service - Handles embedding generation with caching and progress tracking.
"""

import logging
import time
from typing import List, Optional, Callable, Dict, Any
import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from ..models.data_models import DataConfig, EmbeddingConfig, EmbeddingResult, ModelInfo
from .model_management_service import ModelManagementService
from .cache_service import CacheService

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings with caching and optimization."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the embedding service.

        Args:
            cache_dir: Directory for caching embeddings and models
        """
        self.model_service = ModelManagementService(cache_dir)
        self.cache_service = CacheService(cache_dir)

        logger.info("Embedding service initialized")

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning(
                "SentenceTransformers not available - embedding functionality will be limited"
            )

    def _resolve_device(self, device: str) -> str:
        """Resolve device string, converting 'auto' to actual device.

        Args:
            device: Device string ('auto', 'cpu', 'cuda')

        Returns:
            Resolved device string
        """
        if device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        return device

    def generate_embeddings(
        self,
        data_config: DataConfig,
        embedding_config: EmbeddingConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> EmbeddingResult:
        """Generate embeddings for the configured data.

        Args:
            data_config: Data configuration with selected columns and settings
            embedding_config: Embedding model configuration
            progress_callback: Optional callback for progress updates (current, total, message)

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        start_time = time.time()

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("SentenceTransformers not available")
            return EmbeddingResult()

        if not data_config.is_configured:
            logger.error("Data configuration is not valid")
            return EmbeddingResult()

        if not embedding_config.is_configured:
            logger.error("Embedding configuration is not valid")
            return EmbeddingResult()

        # Extract texts from data
        if progress_callback:
            progress_callback(0, 100, "Preparing text data...")

        texts = self._extract_texts(data_config)
        if not texts:
            logger.error("No texts extracted from data")
            return EmbeddingResult()

        logger.info(f"Extracted {len(texts)} texts for embedding generation")

        # Check cache first
        cache_key = self.cache_service.generate_cache_key(
            data_config,
            embedding_config.model_info,
            {
                "batch_size": embedding_config.batch_size,
                "normalize_embeddings": embedding_config.normalize_embeddings,
            },
        )

        if progress_callback:
            progress_callback(10, 100, "Checking cache...")

        cached_result = self.cache_service.get_cached_embeddings(cache_key)
        if cached_result is not None:
            embeddings, cached_texts = cached_result

            # Validate cached data matches current data
            if len(cached_texts) == len(texts):
                data_hash = self.cache_service.compute_data_hash(texts)
                cache_info = self.cache_service._cache_index.get(cache_key)

                if cache_info and cache_info.data_hash == data_hash:
                    logger.info("Using cached embeddings")

                    if progress_callback:
                        progress_callback(100, 100, "Loaded from cache")

                    return EmbeddingResult(
                        embeddings=embeddings,
                        texts=texts,
                        model_info=embedding_config.model_info,
                        processing_time_seconds=time.time() - start_time,
                        cache_hit=True,
                        cache_info=cache_info,
                    )

        # Generate embeddings
        if progress_callback:
            progress_callback(20, 100, "Loading model...")

        # Ensure model is loaded
        if not embedding_config.model_info.is_loaded:
            success = self.model_service.load_model(embedding_config.model_info)
            if not success:
                logger.error("Failed to load embedding model")
                return EmbeddingResult()

        model = self.model_service.get_loaded_model(
            embedding_config.model_info.model_name
        )
        if model is None:
            logger.error("Model not available after loading")
            return EmbeddingResult()

        if progress_callback:
            progress_callback(30, 100, "Generating embeddings...")

        # Generate embeddings with progress tracking
        embeddings = self._generate_embeddings_with_progress(
            model, texts, embedding_config, progress_callback
        )

        if embeddings is None:
            logger.error("Failed to generate embeddings")
            return EmbeddingResult()

        # Cache the results
        if progress_callback:
            progress_callback(95, 100, "Caching results...")

        data_hash = self.cache_service.compute_data_hash(texts)
        self.cache_service.save_embeddings(
            cache_key, embeddings, texts, embedding_config.model_info, data_hash
        )

        processing_time = time.time() - start_time

        if progress_callback:
            progress_callback(100, 100, f"Completed in {processing_time:.1f}s")

        logger.info(
            f"Generated embeddings: {embeddings.shape} in {processing_time:.2f}s"
        )

        return EmbeddingResult(
            embeddings=embeddings,
            texts=texts,
            model_info=embedding_config.model_info,
            processing_time_seconds=processing_time,
            cache_hit=False,
            batch_stats={
                "total_texts": len(texts),
                "batch_size": embedding_config.batch_size,
                "embedding_dimension": embeddings.shape[1],
                "memory_usage_mb": embeddings.nbytes / (1024 * 1024),
            },
        )

    def _extract_texts(self, data_config: DataConfig) -> List[str]:
        """Extract and combine texts from the data configuration.

        Args:
            data_config: Data configuration

        Returns:
            List of combined texts
        """
        if (
            not data_config.file_metadata
            or data_config.file_metadata.preview_data is None
        ):
            logger.error("No data available for text extraction")
            return []

        # For now, use preview data. In a full implementation, you'd want to load the full dataset
        # This is a simplified version for Phase 3
        df = data_config.file_metadata.preview_data
        texts = []

        for _, row in df.iterrows():
            text_parts = []

            for col in data_config.selected_columns:
                if col in row and pd.notna(row[col]):
                    text = str(row[col]).strip()
                    if text:
                        if data_config.include_column_names:
                            text_parts.append(f"{col}: {text}")
                        else:
                            text_parts.append(text)

            if text_parts:
                combined_text = data_config.text_combination_separator.join(text_parts)

                # Apply text length filters
                if len(combined_text) >= data_config.min_text_length:
                    if (
                        data_config.max_text_length is None
                        or len(combined_text) <= data_config.max_text_length
                    ):
                        texts.append(combined_text)

        # Remove empty texts if configured
        if data_config.remove_empty_rows:
            texts = [text for text in texts if text.strip()]

        return texts

    def _generate_embeddings_with_progress(
        self,
        model: SentenceTransformer,
        texts: List[str],
        config: EmbeddingConfig,
        progress_callback: Optional[Callable[[int, int, str], None]],
    ) -> Optional[np.ndarray]:
        """Generate embeddings with progress tracking.

        Args:
            model: SentenceTransformer model
            texts: List of texts to embed
            config: Embedding configuration
            progress_callback: Progress callback function

        Returns:
            Embeddings array or None if failed
        """
        try:
            total_texts = len(texts)

            if config.batch_size >= total_texts:
                # Single batch
                if progress_callback:
                    progress_callback(50, 100, f"Processing {total_texts} texts...")

                embeddings = model.encode(
                    texts,
                    batch_size=config.batch_size,
                    normalize_embeddings=config.normalize_embeddings,
                    convert_to_tensor=config.convert_to_tensor,
                    device=self._resolve_device(config.device),
                    show_progress_bar=False,  # We handle progress ourselves
                )

                if config.convert_to_tensor:
                    embeddings = embeddings.cpu().numpy()

                return embeddings

            else:
                # Multiple batches with progress tracking
                all_embeddings = []
                batch_size = config.batch_size

                for i in range(0, total_texts, batch_size):
                    batch_end = min(i + batch_size, total_texts)
                    batch_texts = texts[i:batch_end]

                    # Update progress
                    progress_pct = 30 + int(
                        (i / total_texts) * 60
                    )  # 30-90% for embedding generation
                    if progress_callback:
                        progress_callback(
                            progress_pct,
                            100,
                            f"Processing batch {i//batch_size + 1}/{(total_texts + batch_size - 1)//batch_size}",
                        )

                    batch_embeddings = model.encode(
                        batch_texts,
                        batch_size=len(batch_texts),
                        normalize_embeddings=config.normalize_embeddings,
                        convert_to_tensor=config.convert_to_tensor,
                        device=self._resolve_device(config.device),
                        show_progress_bar=False,
                    )

                    if config.convert_to_tensor:
                        batch_embeddings = batch_embeddings.cpu().numpy()

                    all_embeddings.append(batch_embeddings)

                # Combine all batches
                embeddings = np.vstack(all_embeddings)
                return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return None

    def test_model_performance(
        self, model_info: ModelInfo, sample_texts: List[str]
    ) -> Dict[str, Any]:
        """Test model performance on sample texts.

        Args:
            model_info: Model to test
            sample_texts: Sample texts for testing

        Returns:
            Performance metrics
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return {"error": "SentenceTransformers not available"}

        if not sample_texts:
            return {"error": "No sample texts provided"}

        try:
            # Load model if needed
            if not model_info.is_loaded:
                success = self.model_service.load_model(model_info)
                if not success:
                    return {"error": "Failed to load model"}

            model = self.model_service.get_loaded_model(model_info.model_name)
            if model is None:
                return {"error": "Model not available"}

            # Test embedding generation
            start_time = time.time()

            embeddings = model.encode(
                sample_texts[
                    : min(10, len(sample_texts))
                ],  # Test with up to 10 samples
                show_progress_bar=False,
            )

            processing_time = time.time() - start_time

            return {
                "processing_time_seconds": processing_time,
                "texts_per_second": len(sample_texts) / processing_time
                if processing_time > 0
                else 0,
                "embedding_dimension": embeddings.shape[1]
                if hasattr(embeddings, "shape")
                else 0,
                "sample_embedding_shape": embeddings.shape
                if hasattr(embeddings, "shape")
                else None,
                "memory_usage_mb": model_info.memory_usage_mb or 0,
            }

        except Exception as e:
            logger.error(f"Error testing model performance: {str(e)}")
            return {"error": str(e)}

    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available embedding models."""
        return self.model_service.get_available_models()

    def discover_models(self) -> List[ModelInfo]:
        """Discover local models."""
        return self.model_service.discover_local_models()

    def load_model(self, model_info: ModelInfo) -> bool:
        """Load an embedding model."""
        return self.model_service.load_model(model_info)

    def unload_model(self, model_name: str) -> bool:
        """Unload an embedding model."""
        return self.model_service.unload_model(model_name)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        return self.cache_service.get_cache_stats()

    def clear_cache(self) -> bool:
        """Clear embedding cache."""
        return self.cache_service.clear_cache()

    def cleanup_expired_cache(self, max_age_days: int = 30) -> int:
        """Clean up expired cache entries."""
        return self.cache_service.cleanup_expired_cache(max_age_days)

    @property
    def current_model(self):
        """Get the currently loaded model (for representation models)."""
        # Get the most recently loaded model
        logger.debug(
            f"Checking loaded models: {list(self.model_service._loaded_models.keys())}"
        )
        if self.model_service._loaded_models:
            model = next(iter(self.model_service._loaded_models.values()))
            logger.debug(f"Returning current model: {type(model)}")
            return model
        logger.debug("No loaded models found")
        return None

    def get_loaded_model(self, model_name: str):
        """Get a specific loaded model by name.

        Args:
            model_name: Name of the model to retrieve

        Returns:
            Loaded model or None if not found
        """
        return self.model_service.get_loaded_model(model_name)
