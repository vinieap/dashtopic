"""
Data models for hyperparameter optimization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
import numpy as np
from datetime import datetime

from .data_models import TopicResult, TopicModelConfig


class OptimizationStrategy(Enum):
    """Optimization strategy types."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    

class MetricType(Enum):
    """Types of metrics for optimization."""
    # Clustering metrics
    SILHOUETTE = "silhouette_score"
    CALINSKI_HARABASZ = "calinski_harabasz_score"
    DAVIES_BOULDIN = "davies_bouldin_score"
    
    # Topic coherence metrics
    COHERENCE_CV = "c_v"
    COHERENCE_UMASS = "c_umass"
    COHERENCE_UCI = "c_uci"
    COHERENCE_NPMI = "c_npmi"
    
    # Diversity metrics
    TOPIC_DIVERSITY = "topic_diversity"
    TOPIC_COVERAGE = "topic_coverage"
    
    # Performance metrics
    TRAINING_TIME = "training_time"
    MEMORY_USAGE = "memory_usage"
    INFERENCE_TIME = "inference_time"


@dataclass
class ParameterRange:
    """Definition of a parameter search range."""
    name: str
    param_type: str  # "int", "float", "categorical", "bool"
    values: Optional[List[Any]] = None  # For categorical
    min_value: Optional[Union[int, float]] = None  # For numeric
    max_value: Optional[Union[int, float]] = None  # For numeric
    step: Optional[Union[int, float]] = None  # For numeric grid search
    log_scale: bool = False  # For numeric parameters
    
    def get_grid_values(self) -> List[Any]:
        """Get all values for grid search."""
        if self.param_type == "categorical" or self.param_type == "bool":
            return self.values or []
        elif self.param_type in ["int", "float"]:
            if self.log_scale:
                # Generate log-spaced values
                values = np.logspace(
                    np.log10(self.min_value),
                    np.log10(self.max_value),
                    num=int((self.max_value - self.min_value) / (self.step or 1)) + 1
                )
                if self.param_type == "int":
                    values = [int(v) for v in values]
                return list(values)
            else:
                # Generate linear-spaced values
                if self.param_type == "int":
                    return list(range(int(self.min_value), int(self.max_value) + 1, int(self.step or 1)))
                else:
                    values = []
                    current = self.min_value
                    while current <= self.max_value:
                        values.append(current)
                        current += self.step or 0.1
                    return values
        return []


@dataclass
class ParameterSpace:
    """Complete parameter space for optimization."""
    # Clustering parameters
    clustering_algorithm: Optional[ParameterRange] = None
    n_clusters: Optional[ParameterRange] = None  # For k-means
    min_cluster_size: Optional[ParameterRange] = None  # For HDBSCAN
    min_samples: Optional[ParameterRange] = None  # For HDBSCAN
    metric: Optional[ParameterRange] = None  # Distance metric
    
    # UMAP parameters
    n_neighbors: Optional[ParameterRange] = None
    n_components: Optional[ParameterRange] = None
    min_dist: Optional[ParameterRange] = None
    
    # Vectorization parameters
    ngram_range: Optional[ParameterRange] = None
    min_df: Optional[ParameterRange] = None
    max_df: Optional[ParameterRange] = None
    
    # BERTopic parameters
    top_n_words: Optional[ParameterRange] = None
    nr_topics: Optional[ParameterRange] = None
    
    def get_all_parameters(self) -> Dict[str, ParameterRange]:
        """Get all non-None parameters."""
        params = {}
        for name, value in self.__dict__.items():
            if value is not None:
                params[name] = value
        return params


@dataclass
class OptimizationConfig:
    """Configuration for optimization run."""
    strategy: OptimizationStrategy
    parameter_space: ParameterSpace
    metrics: List[MetricType]
    primary_metric: MetricType
    minimize_primary: bool = True  # True for metrics like Davies-Bouldin
    
    # Resource constraints
    max_iterations: Optional[int] = None
    max_time_minutes: Optional[int] = None
    max_memory_gb: Optional[float] = None
    n_jobs: int = 1  # Parallel jobs
    
    # Cross-validation
    cv_folds: int = 3
    cv_strategy: str = "stratified"  # or "random"
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10  # Stop if no improvement for N iterations
    min_delta: float = 0.001  # Minimum improvement to continue
    
    # Sampling
    sample_size: Optional[int] = None  # Use subset for faster optimization
    
    @property
    def total_combinations(self) -> int:
        """Calculate total parameter combinations for grid search."""
        if self.strategy != OptimizationStrategy.GRID_SEARCH:
            return -1
        
        total = 1
        for param in self.parameter_space.get_all_parameters().values():
            total *= len(param.get_grid_values())
        return total


@dataclass
class MetricResult:
    """Result for a single metric evaluation."""
    metric_type: MetricType
    value: float
    std_dev: Optional[float] = None  # From cross-validation
    computation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationRun:
    """Single optimization run with parameters and results."""
    run_id: str
    iteration: int
    parameters: Dict[str, Any]
    metrics: Dict[MetricType, MetricResult]
    topic_result: Optional[TopicResult] = None
    
    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # Resources
    memory_usage_mb: float = 0.0
    
    # Status
    status: str = "pending"  # pending, running, completed, failed
    error_message: Optional[str] = None
    
    @property
    def duration_seconds(self) -> float:
        """Get run duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def get_metric_value(self, metric_type: MetricType) -> Optional[float]:
        """Get value for a specific metric."""
        if metric_type in self.metrics:
            return self.metrics[metric_type].value
        return None


@dataclass
class OptimizationResult:
    """Complete optimization result."""
    optimization_id: str
    config: OptimizationConfig
    runs: List[OptimizationRun] = field(default_factory=list)
    best_run: Optional[OptimizationRun] = None
    
    # Summary statistics
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_iterations: int = 0
    completed_iterations: int = 0
    failed_iterations: int = 0
    
    # Early stopping info
    stopped_early: bool = False
    stopping_reason: Optional[str] = None
    
    def add_run(self, run: OptimizationRun):
        """Add a run and update best if needed."""
        self.runs.append(run)
        self.total_iterations += 1
        
        if run.status == "completed":
            self.completed_iterations += 1
            
            # Update best run
            if self.best_run is None:
                self.best_run = run
            else:
                primary_metric = self.config.primary_metric
                current_value = run.get_metric_value(primary_metric)
                best_value = self.best_run.get_metric_value(primary_metric)
                
                if current_value is not None and best_value is not None:
                    if self.config.minimize_primary:
                        if current_value < best_value:
                            self.best_run = run
                    else:
                        if current_value > best_value:
                            self.best_run = run
        
        elif run.status == "failed":
            self.failed_iterations += 1
    
    def get_top_runs(self, n: int = 10, metric: Optional[MetricType] = None) -> List[OptimizationRun]:
        """Get top N runs by metric."""
        if metric is None:
            metric = self.config.primary_metric
        
        completed_runs = [r for r in self.runs if r.status == "completed"]
        
        # Sort by metric
        def sort_key(run):
            value = run.get_metric_value(metric)
            return value if value is not None else float('inf')
        
        reverse = not self.config.minimize_primary if metric == self.config.primary_metric else True
        sorted_runs = sorted(completed_runs, key=sort_key, reverse=reverse)
        
        return sorted_runs[:n]
    
    @property
    def duration_seconds(self) -> float:
        """Get total optimization duration."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


@dataclass
class ComparisonReport:
    """Report comparing multiple optimization runs."""
    runs: List[OptimizationRun]
    metrics_summary: Dict[MetricType, Dict[str, float]]  # min, max, mean, std
    parameter_importance: Dict[str, float]  # Parameter -> importance score
    best_parameters: Dict[str, Any]
    
    # Visualizations data
    parallel_coords_data: Optional[Dict[str, Any]] = None
    metric_history: Optional[Dict[MetricType, List[float]]] = None
    parameter_heatmap: Optional[np.ndarray] = None