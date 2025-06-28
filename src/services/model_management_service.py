"""
Model Management Service - Handles discovery and management of embedding models.
"""
import logging
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import time
import sys
import gc

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    torch = None

from ..models.data_models import ModelInfo

logger = logging.getLogger(__name__)


class ModelManagementService:
    """Service for managing embedding models."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the model management service.
        
        Args:
            cache_dir: Directory for caching model information
        """
        self.cache_dir = Path(cache_dir or Path.home() / ".bertopic_app" / "models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._loaded_models: Dict[str, SentenceTransformer] = {}
        self._model_cache_file = self.cache_dir / "model_cache.json"
        self._discovered_models: List[ModelInfo] = []
        
        logger.info(f"Model management service initialized with cache dir: {self.cache_dir}")
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("SentenceTransformers not available - model functionality will be limited")
    
    def discover_local_models(self) -> List[ModelInfo]:
        """Discover locally available models."""
        logger.info("Discovering local models...")
        models = []
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("SentenceTransformers not available - cannot discover models")
            return models
        
        # Check common SentenceTransformers cache locations
        st_cache_dirs = [
            Path.home() / ".cache" / "torch" / "sentence_transformers",
            Path.home() / ".cache" / "huggingface" / "transformers",
            self.cache_dir / "sentence_transformers"
        ]
        
        for cache_dir in st_cache_dirs:
            if cache_dir.exists():
                models.extend(self._scan_directory_for_models(cache_dir))
        
        # Load cached model information
        cached_models = self._load_cached_model_info()
        
        # Merge discovered and cached models
        model_dict = {m.model_name: m for m in models}
        for cached_model in cached_models:
            if cached_model.model_name not in model_dict:
                # Check if cached model path still exists
                if Path(cached_model.model_path).exists():
                    model_dict[cached_model.model_name] = cached_model
        
        self._discovered_models = list(model_dict.values())
        
        # Save updated cache
        self._save_cached_model_info(self._discovered_models)
        
        logger.info(f"Discovered {len(self._discovered_models)} local models")
        return self._discovered_models.copy()
    
    def get_popular_models(self) -> List[ModelInfo]:
        """Get list of popular pre-trained models that can be downloaded."""
        popular_models = [
            ModelInfo(
                model_name="all-MiniLM-L6-v2",
                model_path="sentence-transformers/all-MiniLM-L6-v2",
                model_type="sentence-transformers",
                embedding_dimension=384,
                max_sequence_length=256,
                description="Fast and efficient model for general-purpose embeddings",
                model_size_mb=80,
                supports_languages=["en"]
            ),
            ModelInfo(
                model_name="all-mpnet-base-v2",
                model_path="sentence-transformers/all-mpnet-base-v2",
                model_type="sentence-transformers",
                embedding_dimension=768,
                max_sequence_length=384,
                description="High-quality embeddings for semantic similarity",
                model_size_mb=420,
                supports_languages=["en"]
            ),
            ModelInfo(
                model_name="paraphrase-multilingual-MiniLM-L12-v2",
                model_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_type="sentence-transformers",
                embedding_dimension=384,
                max_sequence_length=128,
                description="Multilingual model for 50+ languages",
                model_size_mb=420,
                supports_languages=["multilingual"]
            ),
            ModelInfo(
                model_name="distilbert-base-nli-stsb-mean-tokens",
                model_path="sentence-transformers/distilbert-base-nli-stsb-mean-tokens",
                model_type="sentence-transformers",
                embedding_dimension=768,
                max_sequence_length=128,
                description="Distilled BERT model for semantic similarity",
                model_size_mb=250,
                supports_languages=["en"]
            )
        ]
        
        # Mark as loaded if they exist locally
        local_model_names = {m.model_name for m in self._discovered_models}
        for model in popular_models:
            if model.model_name in local_model_names:
                model.is_loaded = True
        
        return popular_models
    
    def load_model(self, model_info: ModelInfo, force_reload: bool = False) -> bool:
        """Load a model into memory.
        
        Args:
            model_info: Information about the model to load
            force_reload: Force reload even if already loaded
            
        Returns:
            True if model loaded successfully
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("SentenceTransformers not available - cannot load model")
            return False
        
        model_key = model_info.model_name
        
        # Check if already loaded
        if model_key in self._loaded_models and not force_reload:
            logger.info(f"Model {model_key} already loaded")
            model_info.is_loaded = True
            return True
        
        try:
            logger.info(f"Loading model: {model_key}")
            start_time = time.time()
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load the model
            model = SentenceTransformer(model_info.model_path, device=device)
            
            load_time = time.time() - start_time
            
            # Update model info
            model_info.is_loaded = True
            model_info.load_time_seconds = load_time
            model_info.embedding_dimension = model.get_sentence_embedding_dimension()
            model_info.max_sequence_length = model.get_max_seq_length()
            
            # Estimate memory usage
            if torch.cuda.is_available():
                model_info.memory_usage_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            else:
                # Rough estimation for CPU
                model_info.memory_usage_mb = sys.getsizeof(model) / (1024 * 1024)
            
            # Store loaded model
            self._loaded_models[model_key] = model
            
            logger.info(f"Model {model_key} loaded successfully in {load_time:.2f}s")
            logger.info(f"Embedding dimension: {model_info.embedding_dimension}")
            logger.info(f"Max sequence length: {model_info.max_sequence_length}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {str(e)}")
            model_info.is_loaded = False
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            True if model unloaded successfully
        """
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            
            # Find and update model info
            for model_info in self._discovered_models:
                if model_info.model_name == model_name:
                    model_info.is_loaded = False
                    model_info.memory_usage_mb = None
                    break
            
            # Force garbage collection
            gc.collect()
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Model {model_name} unloaded successfully")
            return True
        
        logger.warning(f"Model {model_name} was not loaded")
        return False
    
    def get_loaded_model(self, model_name: str) -> Optional[SentenceTransformer]:
        """Get a loaded model instance.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Loaded model instance or None
        """
        return self._loaded_models.get(model_name)
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information or None
        """
        for model_info in self._discovered_models:
            if model_info.model_name == model_name:
                return model_info
        return None
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of all available models (local + popular)."""
        local_models = self._discovered_models.copy()
        popular_models = self.get_popular_models()
        
        # Merge, avoiding duplicates
        model_dict = {m.model_name: m for m in local_models}
        for popular_model in popular_models:
            if popular_model.model_name not in model_dict:
                model_dict[popular_model.model_name] = popular_model
        
        return list(model_dict.values())
    
    def _scan_directory_for_models(self, directory: Path) -> List[ModelInfo]:
        """Scan a directory for SentenceTransformer models."""
        models = []
        
        try:
            for item in directory.iterdir():
                if item.is_dir():
                    # Check if directory contains model files
                    if self._is_sentence_transformer_model(item):
                        try:
                            model_info = self._create_model_info_from_path(item)
                            if model_info:
                                models.append(model_info)
                        except Exception as e:
                            logger.warning(f"Error processing model directory {item}: {str(e)}")
                    else:
                        # Recursively scan subdirectories
                        models.extend(self._scan_directory_for_models(item))
        except PermissionError:
            logger.warning(f"Permission denied accessing directory: {directory}")
        except Exception as e:
            logger.warning(f"Error scanning directory {directory}: {str(e)}")
        
        return models
    
    def _is_sentence_transformer_model(self, path: Path) -> bool:
        """Check if a directory contains a SentenceTransformer model."""
        required_files = ["config.json", "pytorch_model.bin"]
        optional_files = ["modules.json", "sentence_bert_config.json"]
        
        # Check for required files
        for file_name in required_files:
            if not (path / file_name).exists():
                # Also check for alternative pytorch model file
                if file_name == "pytorch_model.bin" and not (path / "model.safetensors").exists():
                    return False
        
        return True
    
    def _create_model_info_from_path(self, path: Path) -> Optional[ModelInfo]:
        """Create ModelInfo from a model directory path."""
        try:
            model_name = path.name
            
            # Try to read config for more information
            config_path = path / "config.json"
            description = None
            max_seq_length = None
            
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        max_seq_length = config.get("max_position_embeddings")
                        description = config.get("_name_or_path", "")
                except Exception as e:
                    logger.debug(f"Could not read config for {model_name}: {str(e)}")
            
            # Calculate directory size
            model_size_mb = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / (1024 * 1024)
            
            return ModelInfo(
                model_name=model_name,
                model_path=str(path),
                model_type="sentence-transformers",
                max_sequence_length=max_seq_length,
                description=description,
                model_size_mb=model_size_mb
            )
            
        except Exception as e:
            logger.warning(f"Error creating model info for {path}: {str(e)}")
            return None
    
    def _load_cached_model_info(self) -> List[ModelInfo]:
        """Load cached model information."""
        if not self._model_cache_file.exists():
            return []
        
        try:
            with open(self._model_cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                models = []
                for item in data.get("models", []):
                    model_info = ModelInfo(**item)
                    models.append(model_info)
                return models
        except Exception as e:
            logger.warning(f"Error loading cached model info: {str(e)}")
            return []
    
    def _save_cached_model_info(self, models: List[ModelInfo]):
        """Save model information to cache."""
        try:
            data = {
                "models": [
                    {
                        "model_name": m.model_name,
                        "model_path": m.model_path,
                        "model_type": m.model_type,
                        "embedding_dimension": m.embedding_dimension,
                        "max_sequence_length": m.max_sequence_length,
                        "description": m.description,
                        "model_size_mb": m.model_size_mb
                    }
                    for m in models
                ]
            }
            
            with open(self._model_cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Error saving cached model info: {str(e)}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage of loaded models."""
        total_memory = 0.0
        model_memory = {}
        
        for model_name, model in self._loaded_models.items():
            model_info = self.get_model_info(model_name)
            if model_info and model_info.memory_usage_mb:
                model_memory[model_name] = model_info.memory_usage_mb
                total_memory += model_info.memory_usage_mb
        
        return {
            "total_mb": total_memory,
            "models": model_memory
        } 