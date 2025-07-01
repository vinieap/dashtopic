"""
Cache Service - Intelligent caching system for embeddings.
"""
import logging
import pickle
import hashlib
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import shutil

from ..models.data_models import CacheInfo, ModelInfo, DataConfig
from ..utils.compressed_cache import CompressedCache, CompressionMethod
from ..utils.memory_manager import MemoryOptimizer, get_memory_stats

logger = logging.getLogger(__name__)


class CacheService:
    """Service for caching embeddings and managing cache lifecycle."""
    
    def __init__(self, cache_dir: Optional[str] = None, max_cache_size_gb: float = 5.0,
                 enable_compression: bool = True, compression_method: CompressionMethod = CompressionMethod.LZ4):
        """Initialize the cache service.
        
        Args:
            cache_dir: Directory for storing cache files
            max_cache_size_gb: Maximum cache size in gigabytes
            enable_compression: Whether to enable compressed caching
            compression_method: Default compression method
        """
        self.cache_dir = Path(cache_dir or Path.home() / ".bertopic_app" / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024 * 1024 * 1024)
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self._cache_index: Dict[str, CacheInfo] = {}
        
        # Initialize compressed cache
        self.enable_compression = enable_compression
        if enable_compression:
            # Reserve 80% of cache size for compressed cache (in-memory + disk)
            compressed_cache_mb = max_cache_size_gb * 1024 * 0.8
            self.compressed_cache = CompressedCache(
                max_memory_mb=compressed_cache_mb * 0.3,  # 30% in memory
                default_compression=compression_method,
                enable_disk_cache=True,
                disk_cache_dir=str(self.cache_dir / "compressed"),
                auto_optimize=True
            )
        else:
            self.compressed_cache = None
        
        logger.info(f"Cache service initialized with cache dir: {self.cache_dir}")
        logger.info(f"Max cache size: {max_cache_size_gb:.1f} GB")
        logger.info(f"Compression enabled: {enable_compression}")
        
        # Load existing cache index
        self._load_cache_index()
        self._validate_cache_files()
    
    def generate_cache_key(self, data_config: DataConfig, model_info: ModelInfo, 
                          additional_params: Optional[Dict[str, Any]] = None) -> str:
        """Generate a unique cache key for the given configuration.
        
        Args:
            data_config: Data configuration
            model_info: Model information  
            additional_params: Additional parameters to include in key
            
        Returns:
            Unique cache key string
        """
        # Create hash components
        hash_components = {
            "model_name": model_info.model_name,
            "model_path": model_info.model_path,
            "selected_columns": sorted(data_config.selected_columns),
            "text_combination_method": data_config.text_combination_method,
            "text_combination_separator": data_config.text_combination_separator,
            "include_column_names": data_config.include_column_names,
            "remove_empty_rows": data_config.remove_empty_rows,
            "min_text_length": data_config.min_text_length,
            "max_text_length": data_config.max_text_length,
            "preprocessing_steps": data_config.preprocessing_steps
        }
        
        # Add file-specific information
        if data_config.file_metadata:
            hash_components.update({
                "file_path": data_config.file_metadata.file_path,
                "file_size": data_config.file_metadata.file_size_bytes,
                "row_count": data_config.file_metadata.row_count,
                "column_count": data_config.file_metadata.column_count
            })
        
        # Add additional parameters
        if additional_params:
            hash_components.update(additional_params)
        
        # Create deterministic hash
        hash_string = json.dumps(hash_components, sort_keys=True, ensure_ascii=True)
        cache_key = hashlib.sha256(hash_string.encode('utf-8')).hexdigest()[:16]
        
        logger.debug(f"Generated cache key: {cache_key}")
        return cache_key
    
    def get_cached_embeddings(self, cache_key: str) -> Optional[Tuple[np.ndarray, List[str]]]:
        """Retrieve cached embeddings.
        
        Args:
            cache_key: Cache key to lookup
            
        Returns:
            Tuple of (embeddings, texts) if found, None otherwise
        """
        if cache_key not in self._cache_index:
            logger.debug(f"Cache key {cache_key} not found in index")
            return None
        
        cache_info = self._cache_index[cache_key]
        cache_file = Path(cache_info.file_path)
        
        if not cache_file.exists():
            logger.warning(f"Cache file missing: {cache_file}")
            self._remove_from_index(cache_key)
            return None
        
        try:
            logger.info(f"Loading cached embeddings from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            embeddings = cached_data['embeddings']
            texts = cached_data['texts']
            
            # Update access time
            cache_info.last_accessed = datetime.now()
            self._save_cache_index()
            
            logger.info(f"Successfully loaded cached embeddings: {embeddings.shape}")
            return embeddings, texts
            
        except Exception as e:
            logger.error(f"Error loading cached embeddings: {str(e)}")
            self._remove_from_index(cache_key)
            return None
    
    def save_embeddings(self, cache_key: str, embeddings: np.ndarray, texts: List[str],
                       model_info: ModelInfo, data_hash: str) -> bool:
        """Save embeddings to cache.
        
        Args:
            cache_key: Cache key for storage
            embeddings: Embeddings array
            texts: Corresponding texts
            model_info: Model information
            data_hash: Hash of the source data
            
        Returns:
            True if saved successfully
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            # Prepare data for caching
            cache_data = {
                'embeddings': embeddings,
                'texts': texts,
                'model_info': {
                    'model_name': model_info.model_name,
                    'model_path': model_info.model_path,
                    'embedding_dimension': model_info.embedding_dimension
                },
                'creation_time': datetime.now(),
                'data_hash': data_hash
            }
            
            logger.info(f"Saving embeddings to cache: {cache_file}")
            
            # Save to file
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Create cache info
            file_size = cache_file.stat().st_size
            cache_info = CacheInfo(
                cache_key=cache_key,
                file_path=str(cache_file),
                creation_time=datetime.now(),
                last_accessed=datetime.now(),
                file_size_bytes=file_size,
                embedding_shape=embeddings.shape,
                model_name=model_info.model_name,
                data_hash=data_hash
            )
            
            # Add to index
            self._cache_index[cache_key] = cache_info
            self._save_cache_index()
            
            logger.info(f"Cached embeddings saved successfully: {embeddings.shape}, {file_size / (1024*1024):.1f} MB")
            
            # Clean up old cache if needed
            self._cleanup_cache_if_needed()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving embeddings to cache: {str(e)}")
            if cache_file.exists():
                cache_file.unlink()
            return False
    
    def invalidate_cache(self, cache_key: str) -> bool:
        """Invalidate and remove a specific cache entry.
        
        Args:
            cache_key: Cache key to invalidate
            
        Returns:
            True if invalidated successfully
        """
        if cache_key in self._cache_index:
            cache_info = self._cache_index[cache_key]
            cache_file = Path(cache_info.file_path)
            
            try:
                if cache_file.exists():
                    cache_file.unlink()
                    logger.info(f"Removed cache file: {cache_file}")
                
                self._remove_from_index(cache_key)
                logger.info(f"Invalidated cache entry: {cache_key}")
                return True
                
            except Exception as e:
                logger.error(f"Error invalidating cache entry {cache_key}: {str(e)}")
                return False
        
        logger.warning(f"Cache key not found for invalidation: {cache_key}")
        return False
    
    def clear_cache(self) -> bool:
        """Clear all cached embeddings.
        
        Returns:
            True if cleared successfully
        """
        try:
            logger.info("Clearing all cached embeddings")
            
            # Remove all cache files
            for cache_info in self._cache_index.values():
                cache_file = Path(cache_info.file_path)
                if cache_file.exists():
                    cache_file.unlink()
            
            # Clear index
            self._cache_index.clear()
            self._save_cache_index()
            
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_size = 0
        total_files = len(self._cache_index)
        oldest_entry = None
        newest_entry = None
        
        for cache_info in self._cache_index.values():
            total_size += cache_info.file_size_bytes
            
            if oldest_entry is None or cache_info.creation_time < oldest_entry:
                oldest_entry = cache_info.creation_time
            
            if newest_entry is None or cache_info.creation_time > newest_entry:
                newest_entry = cache_info.creation_time
        
        return {
            "total_files": total_files,
            "total_size_mb": total_size / (1024 * 1024),
            "total_size_gb": total_size / (1024 * 1024 * 1024),
            "max_size_gb": self.max_cache_size_bytes / (1024 * 1024 * 1024),
            "usage_percentage": (total_size / self.max_cache_size_bytes) * 100,
            "oldest_entry": oldest_entry,
            "newest_entry": newest_entry,
            "cache_dir": str(self.cache_dir)
        }
    
    def get_cache_entries(self) -> List[CacheInfo]:
        """Get list of all cache entries.
        
        Returns:
            List of cache info objects
        """
        return list(self._cache_index.values())
    
    def cleanup_expired_cache(self, max_age_days: int = 30) -> int:
        """Clean up cache entries older than specified age.
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            Number of entries removed
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        removed_count = 0
        
        cache_keys_to_remove = []
        for cache_key, cache_info in self._cache_index.items():
            if cache_info.creation_time < cutoff_date:
                cache_keys_to_remove.append(cache_key)
        
        for cache_key in cache_keys_to_remove:
            if self.invalidate_cache(cache_key):
                removed_count += 1
        
        logger.info(f"Cleaned up {removed_count} expired cache entries")
        return removed_count
    
    def _load_cache_index(self):
        """Load cache index from file."""
        if not self.cache_index_file.exists():
            logger.info("No existing cache index found")
            return
        
        try:
            with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for cache_key, cache_data in data.items():
                # Convert datetime strings back to datetime objects
                cache_data['creation_time'] = datetime.fromisoformat(cache_data['creation_time'])
                cache_data['last_accessed'] = datetime.fromisoformat(cache_data['last_accessed'])
                cache_data['embedding_shape'] = tuple(cache_data['embedding_shape'])
                
                self._cache_index[cache_key] = CacheInfo(**cache_data)
            
            logger.info(f"Loaded cache index with {len(self._cache_index)} entries")
            
        except Exception as e:
            logger.error(f"Error loading cache index: {str(e)}")
            self._cache_index = {}
    
    def _save_cache_index(self):
        """Save cache index to file."""
        try:
            data = {}
            for cache_key, cache_info in self._cache_index.items():
                data[cache_key] = {
                    'cache_key': cache_info.cache_key,
                    'file_path': cache_info.file_path,
                    'creation_time': cache_info.creation_time.isoformat(),
                    'last_accessed': cache_info.last_accessed.isoformat(),
                    'file_size_bytes': cache_info.file_size_bytes,
                    'embedding_shape': list(cache_info.embedding_shape),
                    'model_name': cache_info.model_name,
                    'data_hash': cache_info.data_hash,
                    'is_valid': cache_info.is_valid
                }
            
            with open(self.cache_index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving cache index: {str(e)}")
    
    def _validate_cache_files(self):
        """Validate that all indexed cache files exist."""
        invalid_keys = []
        
        for cache_key, cache_info in self._cache_index.items():
            cache_file = Path(cache_info.file_path)
            if not cache_file.exists():
                logger.warning(f"Cache file missing: {cache_file}")
                invalid_keys.append(cache_key)
        
        # Remove invalid entries
        for cache_key in invalid_keys:
            self._remove_from_index(cache_key)
        
        if invalid_keys:
            logger.info(f"Removed {len(invalid_keys)} invalid cache entries")
    
    def _remove_from_index(self, cache_key: str):
        """Remove entry from cache index."""
        if cache_key in self._cache_index:
            del self._cache_index[cache_key]
            self._save_cache_index()
    
    def _cleanup_cache_if_needed(self):
        """Clean up cache if it exceeds size limit."""
        total_size = sum(cache_info.file_size_bytes for cache_info in self._cache_index.values())
        
        if total_size <= self.max_cache_size_bytes:
            return
        
        logger.info(f"Cache size ({total_size / (1024*1024*1024):.2f} GB) exceeds limit, cleaning up...")
        
        # Sort by last accessed time (oldest first)
        sorted_entries = sorted(
            self._cache_index.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Remove oldest entries until under limit
        removed_count = 0
        for cache_key, cache_info in sorted_entries:
            if total_size <= self.max_cache_size_bytes * 0.8:  # Leave some buffer
                break
            
            if self.invalidate_cache(cache_key):
                total_size -= cache_info.file_size_bytes
                removed_count += 1
        
        logger.info(f"Cleaned up {removed_count} cache entries to stay under size limit")
    
    def compute_data_hash(self, texts: List[str]) -> str:
        """Compute hash of text data for cache validation.
        
        Args:
            texts: List of texts to hash
            
        Returns:
            Hash string
        """
        # Create hash of all texts
        combined_text = "".join(texts)
        return hashlib.sha256(combined_text.encode('utf-8')).hexdigest()[:16]
    
    def get_cached_embeddings_fast(self, cache_key: str) -> Optional[Tuple[np.ndarray, List[str]]]:
        """Retrieve cached embeddings using compressed cache (faster).
        
        Args:
            cache_key: Cache key to lookup
            
        Returns:
            Tuple of (embeddings, texts) if found, None otherwise
        """
        if not self.enable_compression or not self.compressed_cache:
            return self.get_cached_embeddings(cache_key)
        
        try:
            # Try compressed cache first
            cached_data = self.compressed_cache.get(cache_key)
            if cached_data is not None:
                logger.info(f"Fast cache hit: {cache_key}")
                return cached_data['embeddings'], cached_data['texts']
            
            # Fallback to regular cache
            logger.debug(f"Fast cache miss, trying regular cache: {cache_key}")
            regular_result = self.get_cached_embeddings(cache_key)
            
            # If found in regular cache, promote to compressed cache
            if regular_result is not None:
                embeddings, texts = regular_result
                self._promote_to_compressed_cache(cache_key, embeddings, texts)
            
            return regular_result
            
        except Exception as e:
            logger.error(f"Error in fast cache retrieval: {e}")
            return self.get_cached_embeddings(cache_key)
    
    def save_embeddings_compressed(self, cache_key: str, embeddings: np.ndarray, texts: List[str],
                                 model_info: ModelInfo, data_hash: str, 
                                 compression: Optional[CompressionMethod] = None) -> bool:
        """Save embeddings using compressed cache.
        
        Args:
            cache_key: Cache key for storage
            embeddings: Embeddings array
            texts: Corresponding texts
            model_info: Model information
            data_hash: Hash of the source data
            compression: Specific compression method (optional)
            
        Returns:
            True if saved successfully
        """
        if not self.enable_compression or not self.compressed_cache:
            return self.save_embeddings(cache_key, embeddings, texts, model_info, data_hash)
        
        try:
            # Prepare data for compressed caching
            cache_data = {
                'embeddings': embeddings,
                'texts': texts,
                'model_info': {
                    'model_name': model_info.model_name,
                    'model_path': model_info.model_path,
                    'embedding_dimension': model_info.embedding_dimension
                },
                'creation_time': datetime.now(),
                'data_hash': data_hash
            }
            
            # Optimize embeddings memory usage
            optimized_embeddings = MemoryOptimizer.optimize_numpy_array(embeddings)
            cache_data['embeddings'] = optimized_embeddings
            
            logger.info(f"Saving to compressed cache: {cache_key}")
            memory_before = get_memory_stats()
            
            # Save to compressed cache
            success = self.compressed_cache.put(
                cache_key, 
                cache_data, 
                compression=compression,
                ttl_seconds=30*24*3600  # 30 days TTL
            )
            
            if success:
                memory_after = get_memory_stats()
                memory_change = memory_after.process_mb - memory_before.process_mb
                
                logger.info(f"Compressed cache save successful: {cache_key}, "
                           f"memory change: {memory_change:+.1f} MB")
                
                # Also save to regular cache as backup
                self.save_embeddings(cache_key, embeddings, texts, model_info, data_hash)
                
                return True
            else:
                logger.warning(f"Compressed cache save failed, using regular cache: {cache_key}")
                return self.save_embeddings(cache_key, embeddings, texts, model_info, data_hash)
                
        except Exception as e:
            logger.error(f"Error in compressed cache save: {e}")
            return self.save_embeddings(cache_key, embeddings, texts, model_info, data_hash)
    
    def _promote_to_compressed_cache(self, cache_key: str, embeddings: np.ndarray, texts: List[str]):
        """Promote regular cache entry to compressed cache.
        
        Args:
            cache_key: Cache key
            embeddings: Embeddings array
            texts: Text list
        """
        if not self.enable_compression or not self.compressed_cache:
            return
        
        try:
            # Get metadata from regular cache
            if cache_key not in self._cache_index:
                return
            
            cache_info = self._cache_index[cache_key]
            
            # Create data for compressed cache
            cache_data = {
                'embeddings': embeddings,
                'texts': texts,
                'model_info': {'model_name': cache_info.model_name},
                'creation_time': cache_info.creation_time,
                'data_hash': cache_info.data_hash
            }
            
            # Save to compressed cache
            self.compressed_cache.put(cache_key, cache_data, ttl_seconds=30*24*3600)
            logger.debug(f"Promoted to compressed cache: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Error promoting to compressed cache: {e}")
    
    def get_enhanced_cache_stats(self) -> Dict[str, Any]:
        """Get enhanced cache statistics including compression stats.
        
        Returns:
            Dictionary with comprehensive cache statistics
        """
        regular_stats = self.get_cache_stats()
        
        if self.enable_compression and self.compressed_cache:
            compressed_stats = self.compressed_cache.get_stats()
            
            # Combine statistics
            enhanced_stats = {
                **regular_stats,
                'compression_enabled': True,
                'compressed_cache': compressed_stats,
                'total_cache_efficiency': {
                    'regular_cache_mb': regular_stats['total_size_mb'],
                    'compressed_cache_mb': compressed_stats['memory_usage_mb'],
                    'total_memory_mb': regular_stats['total_size_mb'] + compressed_stats['memory_usage_mb'],
                    'compression_savings_mb': compressed_stats.get('total_uncompressed_bytes', 0) / 1024**2 - 
                                            compressed_stats.get('total_compressed_bytes', 0) / 1024**2,
                    'hit_rate_improvement': compressed_stats['hit_rate'] - 
                                          (regular_stats.get('hit_rate', 0) if 'hit_rate' in regular_stats else 0)
                }
            }
        else:
            enhanced_stats = {
                **regular_stats,
                'compression_enabled': False,
                'compressed_cache': None
            }
        
        return enhanced_stats
    
    def optimize_cache(self):
        """Run comprehensive cache optimization."""
        logger.info("Running cache optimization")
        
        # Clean up expired entries in regular cache
        removed_count = self.cleanup_expired_cache()
        logger.info(f"Removed {removed_count} expired regular cache entries")
        
        # Optimize compressed cache
        if self.enable_compression and self.compressed_cache:
            self.compressed_cache.optimize()
            logger.info("Compressed cache optimization completed")
        
        # Run memory optimization
        MemoryOptimizer.force_garbage_collection()
        
        # Log final statistics
        stats = self.get_enhanced_cache_stats()
        logger.info(f"Cache optimization completed. "
                   f"Regular: {stats['total_files']} files, "
                   f"Compressed: {stats['compressed_cache']['total_entries'] if stats['compressed_cache'] else 0} entries")
    
    def clear_all_caches(self):
        """Clear both regular and compressed caches."""
        logger.info("Clearing all caches")
        
        # Clear regular cache
        self.clear_cache()
        
        # Clear compressed cache
        if self.enable_compression and self.compressed_cache:
            self.compressed_cache.clear()
        
        logger.info("All caches cleared") 