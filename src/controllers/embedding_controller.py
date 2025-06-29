"""
Embedding Controller - Coordinates embedding generation workflow.
"""
import logging
from typing import Optional, Callable, List, Dict, Any
import threading

from ..models.data_models import (
    DataConfig, EmbeddingConfig, EmbeddingResult, ModelInfo
)
from ..services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class EmbeddingController:
    """Controller for managing embedding generation workflow."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the embedding controller.
        
        Args:
            cache_dir: Directory for caching
        """
        self.embedding_service = EmbeddingService(cache_dir)
        self._current_task: Optional[threading.Thread] = None
        self._cancel_requested = False
        self.current_result: Optional[EmbeddingResult] = None
        
        logger.info("Embedding controller initialized")
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available embedding models."""
        try:
            return self.embedding_service.get_available_models()
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return []
    
    def discover_local_models(self) -> List[ModelInfo]:
        """Discover locally available models."""
        try:
            return self.embedding_service.discover_models()
        except Exception as e:
            logger.error(f"Error discovering local models: {str(e)}")
            return []
    
    def load_model(self, model_info: ModelInfo) -> bool:
        """Load an embedding model.
        
        Args:
            model_info: Model to load
            
        Returns:
            True if loaded successfully
        """
        try:
            return self.embedding_service.load_model(model_info)
        except Exception as e:
            logger.error(f"Error loading model {model_info.model_name}: {str(e)}")
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """Unload an embedding model.
        
        Args:
            model_name: Name of model to unload
            
        Returns:
            True if unloaded successfully
        """
        try:
            return self.embedding_service.unload_model(model_name)
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {str(e)}")
            return False
    
    def test_model_performance(self, model_info: ModelInfo, sample_texts: List[str]) -> Dict[str, Any]:
        """Test model performance on sample texts.
        
        Args:
            model_info: Model to test
            sample_texts: Sample texts for testing
            
        Returns:
            Performance metrics
        """
        try:
            return self.embedding_service.test_model_performance(model_info, sample_texts)
        except Exception as e:
            logger.error(f"Error testing model performance: {str(e)}")
            return {"error": str(e)}
    
    def generate_embeddings(
        self,
        data_config: DataConfig,
        embedding_config: EmbeddingConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        completion_callback: Optional[Callable[[EmbeddingResult], None]] = None,
        error_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        """Generate embeddings asynchronously.
        
        Args:
            data_config: Data configuration
            embedding_config: Embedding configuration
            progress_callback: Progress update callback
            completion_callback: Completion callback
            error_callback: Error callback
            
        Returns:
            True if task started successfully
        """
        if self._current_task and self._current_task.is_alive():
            logger.warning("Embedding generation already in progress")
            if error_callback:
                error_callback("Embedding generation already in progress")
            return False
        
        try:
            self._cancel_requested = False
            
            def embedding_task():
                try:
                    logger.info("Starting embedding generation task")
                    
                    # Wrapper progress callback to check for cancellation
                    def progress_wrapper(current: int, total: int, message: str):
                        if self._cancel_requested:
                            raise InterruptedError("Task cancelled by user")
                        if progress_callback:
                            progress_callback(current, total, message)
                    
                    result = self.embedding_service.generate_embeddings(
                        data_config, embedding_config, progress_wrapper
                    )
                    
                    if self._cancel_requested:
                        logger.info("Embedding generation cancelled")
                        if error_callback:
                            error_callback("Task cancelled by user")
                        return
                    
                    if result.embeddings is not None:
                        logger.info("Embedding generation completed successfully")
                        self.current_result = result  # Store the result
                        if completion_callback:
                            completion_callback(result)
                    else:
                        logger.error("Embedding generation failed")
                        if error_callback:
                            error_callback("Failed to generate embeddings")
                
                except InterruptedError as e:
                    logger.info(f"Embedding generation interrupted: {str(e)}")
                    if error_callback:
                        error_callback(str(e))
                
                except Exception as e:
                    logger.error(f"Error in embedding generation task: {str(e)}")
                    if error_callback:
                        error_callback(f"Error generating embeddings: {str(e)}")
            
            self._current_task = threading.Thread(target=embedding_task, daemon=True)
            self._current_task.start()
            
            logger.info("Embedding generation task started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting embedding generation: {str(e)}")
            if error_callback:
                error_callback(f"Failed to start embedding generation: {str(e)}")
            return False
    
    def cancel_embedding_generation(self) -> bool:
        """Cancel ongoing embedding generation.
        
        Returns:
            True if cancellation requested successfully
        """
        if self._current_task and self._current_task.is_alive():
            logger.info("Requesting cancellation of embedding generation")
            self._cancel_requested = True
            return True
        
        logger.warning("No embedding generation task to cancel")
        return False
    
    def is_embedding_in_progress(self) -> bool:
        """Check if embedding generation is currently in progress.
        
        Returns:
            True if embedding generation is running
        """
        return self._current_task is not None and self._current_task.is_alive()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics.
        
        Returns:
            Cache statistics
        """
        try:
            return self.embedding_service.get_cache_stats()
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {"error": str(e)}
    
    def clear_cache(self) -> bool:
        """Clear embedding cache.
        
        Returns:
            True if cache cleared successfully
        """
        try:
            return self.embedding_service.clear_cache()
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    def cleanup_expired_cache(self, max_age_days: int = 30) -> int:
        """Clean up expired cache entries.
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            Number of entries removed
        """
        try:
            return self.embedding_service.cleanup_expired_cache(max_age_days)
        except Exception as e:
            logger.error(f"Error cleaning up cache: {str(e)}")
            return 0
    
    def validate_embedding_config(self, embedding_config: EmbeddingConfig) -> List[str]:
        """Validate embedding configuration.
        
        Args:
            embedding_config: Configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not embedding_config.model_info:
            errors.append("No model selected")
        elif not embedding_config.model_info.is_loaded:
            errors.append("Selected model is not loaded")
        
        if embedding_config.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        if embedding_config.max_length is not None and embedding_config.max_length <= 0:
            errors.append("Max length must be positive if specified")
        
        return errors
    
    def get_model_recommendations(self, data_size: int, available_memory_gb: float) -> List[str]:
        """Get model recommendations based on data size and available resources.
        
        Args:
            data_size: Number of documents to process
            available_memory_gb: Available memory in GB
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if data_size < 1000:
            recommendations.append("For small datasets, any model will work well")
            recommendations.append("Consider 'all-mpnet-base-v2' for highest quality")
        elif data_size < 10000:
            recommendations.append("For medium datasets, balance speed and quality")
            recommendations.append("Consider 'all-MiniLM-L6-v2' for good speed/quality balance")
        else:
            recommendations.append("For large datasets, prioritize speed")
            recommendations.append("Consider 'all-MiniLM-L6-v2' or similar lightweight models")
        
        if available_memory_gb < 4:
            recommendations.append("Limited memory: Use smaller models like 'all-MiniLM-L6-v2'")
        elif available_memory_gb < 8:
            recommendations.append("Moderate memory: Most models will work well")
        else:
            recommendations.append("Plenty of memory: You can use larger, higher-quality models")
        
        return recommendations
    
    def estimate_processing_time(self, data_size: int, model_info: ModelInfo) -> Dict[str, float]:
        """Estimate processing time for embedding generation.
        
        Args:
            data_size: Number of documents
            model_info: Selected model
            
        Returns:
            Time estimates in seconds
        """
        # These are rough estimates based on typical performance
        # In practice, you'd want to benchmark on the actual hardware
        
        base_time_per_doc = 0.01  # seconds per document (baseline)
        
        if model_info.model_name:
            if "MiniLM" in model_info.model_name:
                multiplier = 0.5  # Faster model
            elif "mpnet" in model_info.model_name:
                multiplier = 1.5  # Slower but higher quality
            elif "distilbert" in model_info.model_name:
                multiplier = 1.0  # Medium speed
            else:
                multiplier = 1.0  # Default
        else:
            multiplier = 1.0
        
        estimated_time = data_size * base_time_per_doc * multiplier
        
        return {
            "estimated_seconds": estimated_time,
            "estimated_minutes": estimated_time / 60,
            "estimated_hours": estimated_time / 3600
        }
    
    def has_embeddings(self) -> bool:
        """Check if embeddings have been generated.
        
        Returns:
            True if embeddings are available
        """
        return self.current_result is not None and self.current_result.embeddings is not None
    
    def get_embeddings(self):
        """Get the current embeddings array.
        
        Returns:
            Embeddings array or None if not available
        """
        if self.current_result and self.current_result.embeddings is not None:
            return self.current_result.embeddings
        return None 