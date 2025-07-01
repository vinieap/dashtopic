"""
Hyperparameter Optimization Controller - Manages optimization workflow and coordination.
"""

import logging
from typing import Optional, Callable, Dict, Any, List, Tuple
import threading
from dataclasses import dataclass
import numpy as np

from ..models import (
    OptimizationConfig,
    OptimizationResult,
    OptimizationRun,
    MetricType,
    ParameterSpace,
    ParameterRange,
    ComparisonReport,
)
from ..services import HyperparameterOptimizationService
from .topic_modeling_controller import TopicModelingController
from .embedding_controller import EmbeddingController
from .data_controller import DataController

logger = logging.getLogger(__name__)


@dataclass
class OptimizationCallbacks:
    """Callbacks for optimization events."""
    on_progress: Optional[Callable[[int, int, str], None]] = None
    on_run_complete: Optional[Callable[[OptimizationRun], None]] = None
    on_optimization_complete: Optional[Callable[[OptimizationResult], None]] = None
    on_error: Optional[Callable[[str], None]] = None


class OptimizationController:
    """Controller for hyperparameter optimization workflow."""
    
    def __init__(
        self,
        optimization_service: HyperparameterOptimizationService,
        topic_controller: TopicModelingController,
        embedding_controller: EmbeddingController,
        data_controller: DataController
    ):
        """Initialize the optimization controller.
        
        Args:
            optimization_service: Hyperparameter optimization service
            topic_controller: Topic modeling controller
            embedding_controller: Embedding controller
            data_controller: Data controller
        """
        self.optimization_service = optimization_service
        self.topic_controller = topic_controller
        self.embedding_controller = embedding_controller
        self.data_controller = data_controller
        
        # State
        self.current_config: Optional[OptimizationConfig] = None
        self.current_result: Optional[OptimizationResult] = None
        self.optimization_history: List[OptimizationResult] = []
        
        # Threading
        self._optimization_thread: Optional[threading.Thread] = None
        self._callbacks: Optional[OptimizationCallbacks] = None
        
        logger.info("Optimization controller initialized")
    
    def set_callbacks(self, callbacks: OptimizationCallbacks):
        """Set callbacks for optimization events."""
        self._callbacks = callbacks
    
    def create_default_parameter_space(self) -> ParameterSpace:
        """Create a default parameter space for optimization."""
        return ParameterSpace(
            # Clustering parameters
            min_cluster_size=ParameterRange(
                name="min_cluster_size",
                param_type="int",
                min_value=5,
                max_value=50,
                step=5
            ),
            min_samples=ParameterRange(
                name="min_samples",
                param_type="int",
                min_value=1,
                max_value=25,
                step=2
            ),
            
            # UMAP parameters
            n_neighbors=ParameterRange(
                name="n_neighbors",
                param_type="int",
                min_value=5,
                max_value=50,
                step=5
            ),
            min_dist=ParameterRange(
                name="min_dist",
                param_type="float",
                min_value=0.0,
                max_value=0.5,
                step=0.05
            ),
            
            # Vectorization parameters
            min_df=ParameterRange(
                name="min_df",
                param_type="int",
                min_value=1,
                max_value=10,
                step=1
            ),
            
            # BERTopic parameters
            top_n_words=ParameterRange(
                name="top_n_words",
                param_type="int",
                min_value=5,
                max_value=20,
                step=5
            )
        )
    
    def validate_optimization_config(self, config: OptimizationConfig) -> Tuple[bool, str]:
        """Validate optimization configuration.
        
        Args:
            config: Optimization configuration to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if data is ready
        if not self.data_controller.is_ready_for_analysis():
            return False, "Data must be loaded and configured before optimization"
        
        # Check if embeddings are available
        if self.embedding_controller.is_embedding_in_progress():
            return False, "Embedding generation is in progress. Please wait for it to complete."
        
        embeddings = self.embedding_controller.get_embeddings()
        if embeddings is None:
            return False, "Embeddings must be generated before optimization"
        
        # Check parameter space
        if not config.parameter_space.get_all_parameters():
            return False, "Parameter space must contain at least one parameter"
        
        # Check metrics
        if not config.metrics:
            return False, "At least one metric must be selected"
        
        if config.primary_metric not in config.metrics:
            return False, "Primary metric must be in the list of metrics"
        
        # Check resource constraints
        if config.n_jobs > 1:
            import multiprocessing
            max_jobs = multiprocessing.cpu_count()
            if config.n_jobs > max_jobs:
                return False, f"Number of jobs ({config.n_jobs}) exceeds CPU count ({max_jobs})"
        
        return True, ""
    
    def start_optimization(
        self,
        config: OptimizationConfig,
        callbacks: Optional[OptimizationCallbacks] = None
    ) -> bool:
        """Start hyperparameter optimization.
        
        Args:
            config: Optimization configuration
            callbacks: Optional callbacks for events
            
        Returns:
            True if optimization started successfully
        """
        # Validate configuration
        is_valid, error_msg = self.validate_optimization_config(config)
        if not is_valid:
            logger.error(f"Invalid optimization config: {error_msg}")
            if callbacks and callbacks.on_error:
                callbacks.on_error(error_msg)
            return False
        
        # Check if already optimizing
        if self._optimization_thread and self._optimization_thread.is_alive():
            error_msg = "Optimization already in progress"
            logger.error(error_msg)
            if callbacks and callbacks.on_error:
                callbacks.on_error(error_msg)
            return False
        
        # Store configuration and callbacks
        self.current_config = config
        self._callbacks = callbacks or self._callbacks
        
        # Get embeddings and corresponding texts
        embeddings = self.embedding_controller.get_embeddings()
        documents = self.embedding_controller.get_embedding_texts()
        
        if embeddings is None or not documents:
            error_msg = "Failed to retrieve embeddings or corresponding texts"
            logger.error(error_msg)
            if self._callbacks and self._callbacks.on_error:
                self._callbacks.on_error(error_msg)
            return False
        
        # Validate that embeddings and texts match
        if len(embeddings) != len(documents):
            error_msg = f"Embeddings ({len(embeddings)}) and texts ({len(documents)}) length mismatch"
            logger.error(error_msg)
            if self._callbacks and self._callbacks.on_error:
                self._callbacks.on_error(error_msg)
            return False
        
        # Start optimization in background thread
        self._optimization_thread = threading.Thread(
            target=self._run_optimization,
            args=(config, embeddings, documents),
            daemon=True
        )
        self._optimization_thread.start()
        
        logger.info("Optimization started successfully")
        return True
    
    def _run_optimization(
        self,
        config: OptimizationConfig,
        embeddings,
        documents: List[str]
    ):
        """Run optimization in background thread."""
        try:
            # Create progress callback wrapper
            def progress_callback(current: int, total: int, message: str):
                if self._callbacks and self._callbacks.on_progress:
                    self._callbacks.on_progress(current, total, message)
                
                # Check for completed runs
                if self.optimization_service.current_optimization:
                    latest_runs = self.optimization_service.current_optimization.runs
                    if len(latest_runs) > current:
                        latest_run = latest_runs[current - 1]
                        if self._callbacks and self._callbacks.on_run_complete:
                            self._callbacks.on_run_complete(latest_run)
            
            # Run optimization
            result = self.optimization_service.start_optimization(
                config=config,
                embeddings=embeddings,
                documents=documents,
                progress_callback=progress_callback
            )
            
            # Store result
            self.current_result = result
            self.optimization_history.append(result)
            
            # Notify completion
            if self._callbacks and self._callbacks.on_optimization_complete:
                self._callbacks.on_optimization_complete(result)
            
            logger.info(f"Optimization completed: {result.completed_iterations} runs")
            
        except Exception as e:
            error_msg = f"Optimization failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if self._callbacks and self._callbacks.on_error:
                self._callbacks.on_error(error_msg)
    
    def stop_optimization(self):
        """Stop the current optimization."""
        if self.optimization_service.is_optimizing:
            self.optimization_service.stop_optimization()
            logger.info("Optimization stop requested")
    
    def is_optimizing(self) -> bool:
        """Check if optimization is currently running."""
        return bool(self._optimization_thread and self._optimization_thread.is_alive())
    
    def get_current_result(self) -> Optional[OptimizationResult]:
        """Get the current optimization result."""
        return self.current_result
    
    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get all optimization results."""
        return self.optimization_history
    
    def apply_best_parameters(self, optimization_result: Optional[OptimizationResult] = None) -> bool:
        """Apply the best parameters from optimization to topic modeling.
        
        Args:
            optimization_result: Result to use (default: current result)
            
        Returns:
            True if parameters were applied successfully
        """
        result = optimization_result or self.current_result
        if not result or not result.best_run:
            logger.error("No optimization result or best run available")
            return False
        
        try:
            # Get best parameters
            best_params = result.best_run.parameters
            
            # Apply to topic modeling controller
            self.topic_controller.update_config_from_parameters(best_params)
            
            logger.info("Applied best parameters to topic modeling configuration")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply best parameters: {str(e)}")
            return False
    
    def create_comparison_report(
        self,
        optimization_result: Optional[OptimizationResult] = None,
        top_n: int = 10
    ) -> ComparisonReport:
        """Create a comparison report for optimization runs.
        
        Args:
            optimization_result: Result to analyze (default: current result)
            top_n: Number of top runs to include
            
        Returns:
            ComparisonReport with analysis
        """
        result = optimization_result or self.current_result
        if not result:
            return ComparisonReport(runs=[], metrics_summary={}, parameter_importance={}, best_parameters={})
        
        # Get top runs
        top_runs = result.get_top_runs(top_n)
        
        # Calculate metrics summary
        metrics_summary = {}
        for metric in result.config.metrics:
            values = []
            for run in result.runs:
                if run.status == "completed" and metric in run.metrics:
                    values.append(run.metrics[metric].value)
            
            if values:
                metrics_summary[metric] = {
                    "min": float(min(values)),
                    "max": float(max(values)),
                    "mean": float(sum(values) / len(values)),
                    "std": float(np.std(values)) if len(values) > 1 else 0.0
                }
        
        # Calculate parameter importance
        parameter_importance = self.optimization_service.get_parameter_importance(result)
        
        # Get best parameters
        best_parameters = result.best_run.parameters if result.best_run else {}
        
        return ComparisonReport(
            runs=top_runs,
            metrics_summary=metrics_summary,
            parameter_importance=parameter_importance,
            best_parameters=best_parameters
        )
    
    def export_optimization_results(
        self,
        filepath: str,
        optimization_result: Optional[OptimizationResult] = None
    ) -> bool:
        """Export optimization results to file.
        
        Args:
            filepath: Path to save results
            optimization_result: Result to export (default: current result)
            
        Returns:
            True if export successful
        """
        import json
        import pandas as pd
        
        result = optimization_result or self.current_result
        if not result:
            logger.error("No optimization result to export")
            return False
        
        try:
            if filepath.endswith('.json'):
                # Export as JSON
                export_data = {
                    "optimization_id": result.optimization_id,
                    "config": {
                        "strategy": result.config.strategy.value,
                        "metrics": [m.value for m in result.config.metrics],
                        "primary_metric": result.config.primary_metric.value,
                        "total_iterations": result.total_iterations,
                        "completed_iterations": result.completed_iterations,
                    },
                    "best_run": {
                        "parameters": result.best_run.parameters,
                        "metrics": {
                            m.value: v.value 
                            for m, v in result.best_run.metrics.items()
                        }
                    } if result.best_run else None,
                    "duration_seconds": result.duration_seconds,
                }
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
            
            elif filepath.endswith('.csv'):
                # Export as CSV
                rows = []
                for run in result.runs:
                    if run.status == "completed":
                        row = {"run_id": run.run_id, "iteration": run.iteration}
                        row.update(run.parameters)
                        for metric, result in run.metrics.items():
                            row[f"metric_{metric.value}"] = result.value
                        rows.append(row)
                
                df = pd.DataFrame(rows)
                df.to_csv(filepath, index=False)
            
            else:
                logger.error(f"Unsupported export format: {filepath}")
                return False
            
            logger.info(f"Exported optimization results to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export results: {str(e)}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_optimization()
        logger.info("Optimization controller cleaned up")