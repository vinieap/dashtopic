"""
Hyperparameter Optimization Service - Handles parameter search and evaluation.
"""

import logging
import time
import uuid
import gc
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from ..models import (
    OptimizationStrategy,
    MetricType,
    ParameterRange,
    ParameterSpace,
    OptimizationConfig,
    MetricResult,
    OptimizationRun,
    OptimizationResult,
    TopicModelConfig,
    ClusteringConfig,
    VectorizationConfig,
    UMAPConfig,
)
from .bertopic_service import BERTopicService

logger = logging.getLogger(__name__)


class HyperparameterOptimizationService:
    """Service for hyperparameter optimization of topic models."""
    
    def __init__(
        self,
        bertopic_service: BERTopicService,
        cache_dir: Optional[str] = None
    ):
        """Initialize the optimization service.
        
        Args:
            bertopic_service: BERTopic service instance
            cache_dir: Directory for caching optimization results
        """
        self.bertopic_service = bertopic_service
        self.cache_dir = cache_dir
        
        # Optimization state
        self.current_optimization: Optional[OptimizationResult] = None
        self.is_optimizing = False
        self.should_stop = False
        
        logger.info("Hyperparameter optimization service initialized")
    
    def start_optimization(
        self,
        config: OptimizationConfig,
        embeddings: np.ndarray,
        documents: List[str],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> OptimizationResult:
        """Start hyperparameter optimization.
        
        Args:
            config: Optimization configuration
            embeddings: Document embeddings
            documents: Document texts
            progress_callback: Progress callback (current, total, message)
            
        Returns:
            OptimizationResult with all runs and best parameters
        """
        if self.is_optimizing:
            raise RuntimeError("Optimization already in progress")
        
        self.is_optimizing = True
        self.should_stop = False
        
        # Create optimization result
        optimization_id = str(uuid.uuid4())
        self.current_optimization = OptimizationResult(
            optimization_id=optimization_id,
            config=config
        )
        
        try:
            # Validate input data
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            if len(documents) != len(embeddings):
                raise ValueError(f"Documents ({len(documents)}) and embeddings ({len(embeddings)}) length mismatch")
            
            if embeddings.ndim != 2:
                raise ValueError(f"Embeddings must be 2D array, got shape {embeddings.shape}")
            
            logger.info(f"Starting optimization with {len(documents)} documents, embedding shape: {embeddings.shape}")
            
            # Temporarily set debug level for detailed logging
            import logging
            original_level = logger.level
            logger.setLevel(logging.DEBUG)
            
            # Sample data if requested
            if config.sample_size and config.sample_size < len(documents):
                indices = np.random.choice(
                    len(documents), 
                    size=config.sample_size, 
                    replace=False
                )
                embeddings = embeddings[indices]
                documents = [documents[i] for i in indices]
                logger.info(f"Using sample of {config.sample_size} documents for optimization")
            
            # Run optimization based on strategy
            if config.strategy == OptimizationStrategy.GRID_SEARCH:
                self._run_grid_search(
                    config, embeddings, documents, progress_callback
                )
            elif config.strategy == OptimizationStrategy.RANDOM_SEARCH:
                self._run_random_search(
                    config, embeddings, documents, progress_callback
                )
            elif config.strategy == OptimizationStrategy.BAYESIAN:
                self._run_bayesian_optimization(
                    config, embeddings, documents, progress_callback
                )
            
            # Finalize results
            self.current_optimization.end_time = datetime.now()
            
            if self.should_stop:
                self.current_optimization.stopped_early = True
                self.current_optimization.stopping_reason = "User cancelled"
            
            # Restore original log level
            logger.setLevel(original_level)
            
            return self.current_optimization
            
        finally:
            self.is_optimizing = False
    
    def stop_optimization(self):
        """Stop the current optimization."""
        self.should_stop = True
        logger.info("Optimization stop requested")
    
    def _run_grid_search(
        self,
        config: OptimizationConfig,
        embeddings: np.ndarray,
        documents: List[str],
        progress_callback: Optional[Callable]
    ):
        """Run grid search optimization."""
        # Generate all parameter combinations
        param_grid = self._generate_parameter_grid(config.parameter_space)
        total_combinations = len(param_grid)
        
        logger.info(f"Starting grid search with {total_combinations} combinations")
        
        if progress_callback:
            progress_callback(0, total_combinations, "Starting grid search...")
        
        # Track early stopping
        best_score = None
        iterations_without_improvement = 0
        
        # Run evaluations
        if config.n_jobs > 1:
            # Parallel execution
            self._run_parallel_evaluations(
                param_grid, config, embeddings, documents, progress_callback
            )
        else:
            # Sequential execution
            for i, params in enumerate(param_grid):
                if self.should_stop:
                    break
                
                # Create and evaluate run
                run = self._evaluate_parameters(
                    params, config, embeddings, documents, i
                )
                self.current_optimization.add_run(run)
                
                # Check early stopping
                if config.early_stopping and run.status == "completed":
                    current_score = run.get_metric_value(config.primary_metric)
                    if current_score is not None:
                        if best_score is None:
                            best_score = current_score
                        else:
                            improvement = abs(current_score - best_score)
                            if improvement < config.min_delta:
                                iterations_without_improvement += 1
                            else:
                                iterations_without_improvement = 0
                                best_score = current_score
                        
                        if iterations_without_improvement >= config.patience:
                            self.current_optimization.stopped_early = True
                            self.current_optimization.stopping_reason = "Early stopping"
                            logger.info("Early stopping triggered")
                            break
                
                if progress_callback:
                    progress_callback(
                        i + 1, 
                        total_combinations, 
                        f"Evaluated {i + 1}/{total_combinations} combinations"
                    )
    
    def _run_random_search(
        self,
        config: OptimizationConfig,
        embeddings: np.ndarray,
        documents: List[str],
        progress_callback: Optional[Callable]
    ):
        """Run random search optimization."""
        max_iterations = config.max_iterations or 50
        
        logger.info(f"Starting random search with {max_iterations} iterations")
        
        if progress_callback:
            progress_callback(0, max_iterations, "Starting random search...")
        
        for i in range(max_iterations):
            if self.should_stop:
                break
            
            # Sample random parameters
            params = self._sample_random_parameters(config.parameter_space)
            
            # Evaluate
            run = self._evaluate_parameters(
                params, config, embeddings, documents, i
            )
            self.current_optimization.add_run(run)
            
            if progress_callback:
                progress_callback(
                    i + 1,
                    max_iterations,
                    f"Evaluated {i + 1}/{max_iterations} configurations"
                )
    
    def _run_bayesian_optimization(
        self,
        config: OptimizationConfig,
        embeddings: np.ndarray,
        documents: List[str],
        progress_callback: Optional[Callable]
    ):
        """Run Bayesian optimization (placeholder for now)."""
        logger.warning("Bayesian optimization not yet implemented, falling back to random search")
        self._run_random_search(config, embeddings, documents, progress_callback)
    
    def _generate_parameter_grid(self, param_space: ParameterSpace) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        params = param_space.get_all_parameters()
        if not params:
            return [{}]
        
        # Get all possible values for each parameter
        param_values = {}
        for name, param_range in params.items():
            param_values[name] = param_range.get_grid_values()
        
        # Generate all combinations
        combinations = []
        for values in product(*param_values.values()):
            combination = dict(zip(param_values.keys(), values))
            combinations.append(combination)
        
        return combinations
    
    def _sample_random_parameters(self, param_space: ParameterSpace) -> Dict[str, Any]:
        """Sample random parameters from the parameter space."""
        sampled = {}
        
        for name, param_range in param_space.get_all_parameters().items():
            if param_range.param_type == "categorical":
                sampled[name] = np.random.choice(param_range.values)
            elif param_range.param_type == "bool":
                sampled[name] = np.random.choice([True, False])
            elif param_range.param_type == "int":
                if param_range.log_scale:
                    value = np.exp(np.random.uniform(
                        np.log(param_range.min_value),
                        np.log(param_range.max_value)
                    ))
                    sampled[name] = int(value)
                else:
                    sampled[name] = np.random.randint(
                        param_range.min_value,
                        param_range.max_value + 1
                    )
            elif param_range.param_type == "float":
                if param_range.log_scale:
                    sampled[name] = np.exp(np.random.uniform(
                        np.log(param_range.min_value),
                        np.log(param_range.max_value)
                    ))
                else:
                    sampled[name] = np.random.uniform(
                        param_range.min_value,
                        param_range.max_value
                    )
        
        return sampled
    
    def _evaluate_parameters(
        self,
        params: Dict[str, Any],
        config: OptimizationConfig,
        embeddings: np.ndarray,
        documents: List[str],
        iteration: int
    ) -> OptimizationRun:
        """Evaluate a single parameter configuration."""
        run_id = f"{self.current_optimization.optimization_id}_{iteration:04d}"
        run = OptimizationRun(
            run_id=run_id,
            iteration=iteration,
            parameters=params,
            metrics={}
        )
        
        try:
            # Create topic model config from parameters
            logger.debug(f"Run {run_id}: Creating topic config with params: {list(params.keys())}")
            topic_config = self._create_topic_config(params)
            
            # Log config validation details
            logger.debug(f"Run {run_id}: Topic config created")
            logger.debug(f"Run {run_id}: embedding_config exists: {topic_config.embedding_config is not None}")
            if topic_config.embedding_config:
                logger.debug(f"Run {run_id}: model_info exists: {topic_config.embedding_config.model_info is not None}")
                if topic_config.embedding_config.model_info:
                    logger.debug(f"Run {run_id}: model_info.is_loaded: {topic_config.embedding_config.model_info.is_loaded}")
                    logger.debug(f"Run {run_id}: model_info.model_type: {topic_config.embedding_config.model_info.model_type}")
            logger.debug(f"Run {run_id}: is_configured: {topic_config.is_configured}")
            
            # Measure memory before
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Train model
            start_time = time.time()
            
            # Validate embeddings and documents match
            if len(documents) != len(embeddings):
                raise ValueError(f"Mismatch: {len(documents)} documents vs {len(embeddings)} embeddings")
            
            # Ensure embeddings are numpy array
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            # Validate embeddings shape
            if embeddings.ndim != 2:
                raise ValueError(f"Embeddings must be 2D array, got shape {embeddings.shape}")
                
            logger.debug(f"Training with {len(documents)} docs, embeddings shape: {embeddings.shape}")
            
            # Use the service to train the model
            topic_result = self.bertopic_service.train_model(
                texts=documents,
                config=topic_config,
                embeddings=embeddings
            )
            
            training_time = time.time() - start_time
            
            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024
            run.memory_usage_mb = memory_after - memory_before
            
            # Check if topic result is valid
            if topic_result is None:
                raise ValueError("Topic modeling returned None result")
            
            # Store topic result
            run.topic_result = topic_result
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                topic_result, embeddings, documents, config.metrics, training_time
            )
            run.metrics = metrics
            
            run.status = "completed"
            run.end_time = datetime.now()
            
            logger.info(f"Run {run_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Run {run_id} failed: {str(e)}")
            run.status = "failed"
            run.error_message = str(e)
            run.end_time = datetime.now()
        
        # Force garbage collection
        gc.collect()
        
        return run
    
    def _create_topic_config(self, params: Dict[str, Any]) -> TopicModelConfig:
        """Create TopicModelConfig from optimization parameters."""
        # Map parameters to config objects
        clustering_config = ClusteringConfig()
        vectorization_config = VectorizationConfig()
        umap_config = UMAPConfig()
        
        # Create embedding config for precomputed embeddings - no model loading needed
        from ..models.data_models import EmbeddingConfig, ModelInfo
        embedding_config = EmbeddingConfig()
        # Set a dummy model info to satisfy validation - mark as loaded since embeddings are already available
        embedding_config.model_info = ModelInfo(
            model_name="precomputed_embeddings",
            model_path="",  # Empty path to avoid loading
            model_type="precomputed",
            is_loaded=True,  # Mark as loaded since embeddings are precomputed
            description="Precomputed embeddings for optimization"
        )
        
        # Clustering parameters
        if "clustering_algorithm" in params:
            clustering_config.algorithm = params["clustering_algorithm"]
        if "n_clusters" in params:
            clustering_config.n_clusters = params["n_clusters"]
        if "min_cluster_size" in params:
            clustering_config.min_cluster_size = params["min_cluster_size"]
        if "min_samples" in params:
            clustering_config.min_samples = params["min_samples"]
        if "metric" in params:
            clustering_config.metric = params["metric"]
        
        # UMAP parameters
        if "n_neighbors" in params:
            umap_config.n_neighbors = params["n_neighbors"]
        if "n_components" in params:
            umap_config.n_components = params["n_components"]
        if "min_dist" in params:
            umap_config.min_dist = params["min_dist"]
        
        # Vectorization parameters
        if "min_df" in params:
            vectorization_config.min_df = params["min_df"]
        if "max_df" in params:
            vectorization_config.max_df = params["max_df"]
        if "ngram_range" in params:
            vectorization_config.ngram_range = params["ngram_range"]
        
        # Create representation config and disable it for optimization
        from ..models.data_models import RepresentationConfig
        representation_config = RepresentationConfig()
        representation_config.use_representation = False  # Disable to avoid model loading
        
        # BERTopic parameters
        topic_config = TopicModelConfig(
            embedding_config=embedding_config,
            clustering_config=clustering_config,
            vectorization_config=vectorization_config,
            umap_config=umap_config,
            representation_config=representation_config
        )
        
        if "top_n_words" in params:
            topic_config.top_n_words = params["top_n_words"]
        if "nr_topics" in params:
            topic_config.nr_topics = params["nr_topics"]
        
        return topic_config
    
    def _calculate_metrics(
        self,
        topic_result,
        embeddings: np.ndarray,
        documents: List[str],
        metrics: List[MetricType],
        training_time: float
    ) -> Dict[MetricType, MetricResult]:
        """Calculate all requested metrics."""
        results = {}
        
        # Validate topic result
        if not hasattr(topic_result, 'topics') or not hasattr(topic_result, 'topic_info'):
            logger.error("Invalid topic result structure")
            return results
        
        # Get topic assignments
        topic_labels = topic_result.topics
        logger.debug(f"Topic labels type: {type(topic_labels)}, shape: {getattr(topic_labels, 'shape', 'N/A')}")
        logger.debug(f"Topic labels sample: {topic_labels[:5] if hasattr(topic_labels, '__getitem__') else 'N/A'}")
        
        # Ensure topic_labels is a numpy array of integers
        if not isinstance(topic_labels, np.ndarray):
            topic_labels = np.array(topic_labels)
        
        # Convert to integers if needed
        topic_labels = topic_labels.astype(int)
        
        # Filter out outliers for some metrics
        non_outlier_mask = topic_labels != -1
        if np.sum(non_outlier_mask) < 2:
            # Not enough non-outlier points for clustering metrics
            logger.warning("Not enough non-outlier points for clustering metrics")
            return results
        
        logger.debug(f"Non-outlier count: {np.sum(non_outlier_mask)}/{len(topic_labels)}")
        logger.debug(f"Unique topics: {np.unique(topic_labels[non_outlier_mask])}")
        
        # Calculate each metric
        for metric_type in metrics:
            start = time.time()
            
            try:
                if metric_type == MetricType.SILHOUETTE:
                    unique_topics = np.unique(topic_labels[non_outlier_mask])
                    logger.debug(f"Unique topics for silhouette: {unique_topics}")
                    
                    if len(unique_topics) > 1:
                        # Get the filtered data
                        filtered_embeddings = embeddings[non_outlier_mask]
                        filtered_labels = topic_labels[non_outlier_mask]
                        
                        logger.debug(f"Silhouette inputs - embeddings shape: {filtered_embeddings.shape}, labels shape: {filtered_labels.shape}")
                        logger.debug(f"Label range: min={np.min(filtered_labels)}, max={np.max(filtered_labels)}")
                        
                        score = silhouette_score(filtered_embeddings, filtered_labels)
                        results[metric_type] = MetricResult(
                            metric_type=metric_type,
                            value=score,
                            computation_time=time.time() - start
                        )
                        logger.debug(f"Silhouette score calculated: {score}")
                    else:
                        logger.warning(f"Skipping silhouette - only {len(unique_topics)} unique topics")
                
                elif metric_type == MetricType.CALINSKI_HARABASZ:
                    if len(np.unique(topic_labels[non_outlier_mask])) > 1:
                        score = calinski_harabasz_score(
                            embeddings[non_outlier_mask],
                            topic_labels[non_outlier_mask]
                        )
                        results[metric_type] = MetricResult(
                            metric_type=metric_type,
                            value=score,
                            computation_time=time.time() - start
                        )
                
                elif metric_type == MetricType.DAVIES_BOULDIN:
                    if len(np.unique(topic_labels[non_outlier_mask])) > 1:
                        score = davies_bouldin_score(
                            embeddings[non_outlier_mask],
                            topic_labels[non_outlier_mask]
                        )
                        results[metric_type] = MetricResult(
                            metric_type=metric_type,
                            value=score,
                            computation_time=time.time() - start
                        )
                
                elif metric_type == MetricType.TOPIC_DIVERSITY:
                    # Calculate topic diversity (unique words / total words)
                    all_words = set()
                    total_words = 0
                    for topic in topic_result.topic_info:
                        words = [w for w, _ in topic.words[:10]]
                        all_words.update(words)
                        total_words += len(words)
                    
                    diversity = len(all_words) / total_words if total_words > 0 else 0
                    results[metric_type] = MetricResult(
                        metric_type=metric_type,
                        value=diversity,
                        computation_time=time.time() - start
                    )
                
                elif metric_type == MetricType.TRAINING_TIME:
                    results[metric_type] = MetricResult(
                        metric_type=metric_type,
                        value=training_time,
                        computation_time=0
                    )
                
                # Add more metrics as needed
                
            except Exception as e:
                logger.error(f"Failed to calculate {metric_type}: {str(e)}")
        
        return results
    
    def _run_parallel_evaluations(
        self,
        param_grid: List[Dict[str, Any]],
        config: OptimizationConfig,
        embeddings: np.ndarray,
        documents: List[str],
        progress_callback: Optional[Callable]
    ):
        """Run evaluations in parallel."""
        completed = 0
        total = len(param_grid)
        
        with ThreadPoolExecutor(max_workers=config.n_jobs) as executor:
            # Submit all tasks
            future_to_params = {
                executor.submit(
                    self._evaluate_parameters,
                    params, config, embeddings, documents, i
                ): (i, params)
                for i, params in enumerate(param_grid)
            }
            
            # Process completed tasks
            for future in as_completed(future_to_params):
                if self.should_stop:
                    # Cancel remaining tasks
                    for f in future_to_params:
                        f.cancel()
                    break
                
                try:
                    run = future.result()
                    self.current_optimization.add_run(run)
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(
                            completed, total,
                            f"Completed {completed}/{total} evaluations"
                        )
                
                except Exception as e:
                    logger.error(f"Evaluation failed: {str(e)}")
                    completed += 1
    
    def get_parameter_importance(
        self, 
        optimization_result: OptimizationResult,
        metric: Optional[MetricType] = None
    ) -> Dict[str, float]:
        """Calculate parameter importance scores.
        
        Args:
            optimization_result: Completed optimization result
            metric: Metric to use for importance (default: primary metric)
            
        Returns:
            Dictionary of parameter names to importance scores
        """
        if metric is None:
            metric = optimization_result.config.primary_metric
        
        # Get completed runs with the metric
        runs = [
            r for r in optimization_result.runs 
            if r.status == "completed" and metric in r.metrics
        ]
        
        if len(runs) < 2:
            return {}
        
        # Calculate variance contribution for each parameter
        importance = {}
        
        # Get all parameter names
        all_params = set()
        for run in runs:
            all_params.update(run.parameters.keys())
        
        # Calculate importance for each parameter
        metric_values = [r.metrics[metric].value for r in runs]
        metric_variance = np.var(metric_values)
        
        for param in all_params:
            # Get unique values for this parameter
            param_values = {}
            for run in runs:
                if param in run.parameters:
                    value = run.parameters[param]
                    if value not in param_values:
                        param_values[value] = []
                    param_values[value].append(run.metrics[metric].value)
            
            # Calculate variance explained
            if len(param_values) > 1:
                # Calculate mean metric value for each parameter value
                means = [np.mean(values) for values in param_values.values()]
                param_variance = np.var(means)
                
                # Importance is proportion of variance explained
                importance[param] = param_variance / metric_variance if metric_variance > 0 else 0
            else:
                importance[param] = 0.0
        
        # Normalize to sum to 1
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance