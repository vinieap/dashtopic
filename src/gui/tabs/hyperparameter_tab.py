"""
Hyperparameter Optimization Tab - UI for hyperparameter search and optimization.
"""

import logging
from tkinter import ttk, messagebox
import customtkinter as ctk
from typing import Optional, List, Dict, Any
import threading
from datetime import datetime

# Import tooltip utility
try:
    from ..utils.tooltip import add_tooltip, PARAMETER_DESCRIPTIONS
    TOOLTIPS_AVAILABLE = True
except ImportError:
    TOOLTIPS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Import at module level to avoid circular imports
try:
    from ...models import (
        OptimizationStrategy,
        MetricType,
        ParameterRange,
        ParameterSpace,
        OptimizationConfig,
        OptimizationResult,
        OptimizationRun,
    )
    from ...controllers.optimization_controller import OptimizationCallbacks
    MODELS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import optimization models: {e}")
    MODELS_AVAILABLE = False


class HyperparameterTab:
    """Tab for hyperparameter optimization of topic models."""
    
    def __init__(self, parent):
        self.parent = parent
        self.controller = None
        
        # UI state
        self.strategy_var = ctk.StringVar(value="grid_search")
        self.primary_metric_var = ctk.StringVar(value="silhouette_score")
        self.minimize_var = ctk.BooleanVar(value=False)
        self.n_jobs_var = ctk.IntVar(value=1)
        self.cv_folds_var = ctk.IntVar(value=3)
        self.early_stopping_var = ctk.BooleanVar(value=True)
        self.patience_var = ctk.IntVar(value=10)
        
        # Parameter ranges
        self.param_widgets = {}
        self.metric_vars = {}
        
        # Results
        self.current_result = None
        
        self.setup_ui()
        logger.info("Hyperparameter optimization tab initialized")
    
    def set_controller(self, controller):
        """Set the optimization controller."""
        self.controller = controller
        logger.info("Optimization controller set")
    
    def setup_ui(self):
        """Set up the tab user interface."""
        # Main content frame with scrolling
        self.main_canvas = ctk.CTkCanvas(self.parent)
        self.main_scrollbar = ctk.CTkScrollbar(self.parent, command=self.main_canvas.yview)
        self.main_canvas.configure(yscrollcommand=self.main_scrollbar.set)
        
        self.main_scrollbar.pack(side="right", fill="y")
        self.main_canvas.pack(side="left", fill="both", expand=True)
        
        # Create scrollable frame
        self.main_frame = ctk.CTkFrame(self.main_canvas)
        self.main_canvas_frame = self.main_canvas.create_window(
            (0, 0), window=self.main_frame, anchor="nw"
        )
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Hyperparameter Optimization",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.title_label.pack(pady=(10, 20))
        
        if not MODELS_AVAILABLE:
            # Show error message if models not available
            error_label = ctk.CTkLabel(
                self.main_frame,
                text="Optimization models not available.\nPlease check the installation.",
                font=ctk.CTkFont(size=14),
                text_color="red"
            )
            error_label.pack(pady=50)
            return
        
        # Create sections
        self.setup_strategy_section()
        self.setup_parameter_section()
        self.setup_metrics_section()
        self.setup_settings_section()
        self.setup_control_section()
        self.setup_results_section()
        
        # Configure canvas scrolling
        self.main_frame.bind("<Configure>", self._on_frame_configure)
        self.main_canvas.bind("<Configure>", self._on_canvas_configure)
    
    def setup_strategy_section(self):
        """Setup optimization strategy selection."""
        strategy_frame = ctk.CTkFrame(self.main_frame)
        strategy_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            strategy_frame,
            text="Optimization Strategy",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Strategy options
        strategies_frame = ctk.CTkFrame(strategy_frame)
        strategies_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        strategies = [
            ("grid_search", "Grid Search"),
            ("random_search", "Random Search"),
            ("bayesian", "Bayesian (Coming Soon)")
        ]
        
        for strategy_value, strategy_name in strategies:
            radio = ctk.CTkRadioButton(
                strategies_frame,
                text=strategy_name,
                variable=self.strategy_var,
                value=strategy_value,
                command=self._on_strategy_changed
            )
            radio.pack(side="left", padx=10)
            if strategy_value == "bayesian":
                radio.configure(state="disabled")
        
        # Strategy description
        self.strategy_desc = ctk.CTkLabel(
            strategy_frame,
            text="Grid Search: Exhaustive search through all parameter combinations",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.strategy_desc.pack(anchor="w", padx=10, pady=(0, 10))
    
    def setup_parameter_section(self):
        """Setup parameter search space configuration with visual grouping."""
        param_frame = ctk.CTkFrame(self.main_frame)
        param_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            param_frame,
            text="Parameter Search Space",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Description
        desc_label = ctk.CTkLabel(
            param_frame,
            text="Configure parameters to optimize. Hover over parameter names for detailed explanations.",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        desc_label.pack(anchor="w", padx=10, pady=(0, 10))
        
        # Parameter groups with visual separation
        self._setup_clustering_parameters(param_frame)
        self._setup_dimensionality_parameters(param_frame)
        self._setup_vectorization_parameters(param_frame)
        self._setup_topic_parameters(param_frame)
        
        # Total combinations label
        self.combinations_label = ctk.CTkLabel(
            param_frame,
            text="Total combinations: calculating...",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.combinations_label.pack(anchor="w", padx=10, pady=(10, 10))
        
        # Update combinations on parameter change
        self._update_combinations_count()
    
    def _setup_clustering_parameters(self, parent):
        """Setup HDBSCAN clustering parameters."""
        group_frame = ctk.CTkFrame(parent)
        group_frame.pack(fill="x", padx=10, pady=(0, 5))
        
        # Group header
        header_frame = ctk.CTkFrame(group_frame)
        header_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(
            header_frame,
            text="🔗 Clustering Parameters (HDBSCAN)",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#4CAF50"
        ).pack(side="left", padx=10, pady=5)
        
        ctk.CTkLabel(
            header_frame,
            text="Controls how documents are grouped into clusters",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(side="left", padx=(10, 0), pady=5)
        
        # Parameters grid
        params_grid = ctk.CTkFrame(group_frame)
        params_grid.pack(fill="x", padx=5, pady=(0, 5))
        
        self._setup_parameter_headers(params_grid)
        
        parameters = [
            ("min_cluster_size", "Min Cluster Size", 5, 50, 5),
            ("min_samples", "Min Samples", 1, 25, 2),
        ]
        
        self._create_parameter_widgets(params_grid, parameters, 1)
    
    def _setup_dimensionality_parameters(self, parent):
        """Setup UMAP dimensionality reduction parameters."""
        group_frame = ctk.CTkFrame(parent)
        group_frame.pack(fill="x", padx=10, pady=(0, 5))
        
        # Group header
        header_frame = ctk.CTkFrame(group_frame)
        header_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(
            header_frame,
            text="📉 Dimensionality Reduction (UMAP)",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#2196F3"
        ).pack(side="left", padx=10, pady=5)
        
        ctk.CTkLabel(
            header_frame,
            text="Reduces high-dimensional embeddings for clustering",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(side="left", padx=(10, 0), pady=5)
        
        # Parameters grid
        params_grid = ctk.CTkFrame(group_frame)
        params_grid.pack(fill="x", padx=5, pady=(0, 5))
        
        self._setup_parameter_headers(params_grid)
        
        parameters = [
            ("n_neighbors", "N Neighbors", 5, 50, 5),
            ("min_dist", "Min Distance", 0.0, 0.5, 0.05),
        ]
        
        self._create_parameter_widgets(params_grid, parameters, 1)
    
    def _setup_vectorization_parameters(self, parent):
        """Setup text vectorization parameters."""
        group_frame = ctk.CTkFrame(parent)
        group_frame.pack(fill="x", padx=10, pady=(0, 5))
        
        # Group header
        header_frame = ctk.CTkFrame(group_frame)
        header_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(
            header_frame,
            text="📝 Text Vectorization",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#FF9800"
        ).pack(side="left", padx=10, pady=5)
        
        ctk.CTkLabel(
            header_frame,
            text="Controls how text is converted to numerical features",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(side="left", padx=(10, 0), pady=5)
        
        # Parameters grid
        params_grid = ctk.CTkFrame(group_frame)
        params_grid.pack(fill="x", padx=5, pady=(0, 5))
        
        self._setup_parameter_headers(params_grid)
        
        parameters = [
            ("min_df", "Min Doc Frequency", 1, 10, 1),
        ]
        
        self._create_parameter_widgets(params_grid, parameters, 1)
    
    def _setup_topic_parameters(self, parent):
        """Setup BERTopic-specific parameters."""
        group_frame = ctk.CTkFrame(parent)
        group_frame.pack(fill="x", padx=10, pady=(0, 5))
        
        # Group header
        header_frame = ctk.CTkFrame(group_frame)
        header_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(
            header_frame,
            text="🎯 Topic Model Configuration",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#9C27B0"
        ).pack(side="left", padx=10, pady=5)
        
        ctk.CTkLabel(
            header_frame,
            text="Controls topic extraction and representation",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(side="left", padx=(10, 0), pady=5)
        
        # Parameters grid
        params_grid = ctk.CTkFrame(group_frame)
        params_grid.pack(fill="x", padx=5, pady=(0, 5))
        
        self._setup_parameter_headers(params_grid)
        
        parameters = [
            ("top_n_words", "Top N Words", 5, 20, 5),
        ]
        
        self._create_parameter_widgets(params_grid, parameters, 1)
    
    def _setup_parameter_headers(self, grid_frame):
        """Setup column headers for parameter grid."""
        headers = ["Parameter", "Enable", "Min", "Max", "Step"]
        for col, header in enumerate(headers):
            label = ctk.CTkLabel(
                grid_frame, 
                text=header, 
                font=ctk.CTkFont(weight="bold", size=11)
            )
            label.grid(row=0, column=col, padx=5, pady=5, sticky="w")
    
    def _create_parameter_widgets(self, grid_frame, parameters, start_row):
        """Create parameter widgets with tooltips."""
        row = start_row
        for param_name, display_name, default_min, default_max, default_step in parameters:
            # Parameter name with tooltip
            param_label = ctk.CTkLabel(grid_frame, text=display_name)
            param_label.grid(row=row, column=0, padx=5, pady=2, sticky="w")
            
            # Add tooltip if available
            if TOOLTIPS_AVAILABLE and param_name in PARAMETER_DESCRIPTIONS:
                add_tooltip(param_label, PARAMETER_DESCRIPTIONS[param_name])
            
            # Enable checkbox
            enabled_var = ctk.BooleanVar(value=True)  # Default to enabled
            enabled_check = ctk.CTkCheckBox(
                grid_frame, text="", variable=enabled_var, width=20
            )
            enabled_check.grid(row=row, column=1, padx=5, pady=2)
            
            # Min value
            min_entry = ctk.CTkEntry(grid_frame, width=60)
            min_entry.insert(0, str(default_min))
            min_entry.grid(row=row, column=2, padx=5, pady=2)
            
            # Max value
            max_entry = ctk.CTkEntry(grid_frame, width=60)
            max_entry.insert(0, str(default_max))
            max_entry.grid(row=row, column=3, padx=5, pady=2)
            
            # Step value
            step_entry = ctk.CTkEntry(grid_frame, width=60)
            step_entry.insert(0, str(default_step))
            step_entry.grid(row=row, column=4, padx=5, pady=2)
            
            # Store widgets
            self.param_widgets[param_name] = {
                "enabled": enabled_var,
                "min": min_entry,
                "max": max_entry,
                "step": step_entry,
                "type": "float" if param_name in ["min_dist"] else "int"
            }
            
            # Bind events for live updates
            enabled_var.trace_add("write", lambda *args: self._update_combinations_count())
            min_entry.bind("<KeyRelease>", lambda e: self._update_combinations_count())
            max_entry.bind("<KeyRelease>", lambda e: self._update_combinations_count())
            step_entry.bind("<KeyRelease>", lambda e: self._update_combinations_count())
            min_entry.bind("<FocusOut>", lambda e: self._update_combinations_count())
            max_entry.bind("<FocusOut>", lambda e: self._update_combinations_count())
            step_entry.bind("<FocusOut>", lambda e: self._update_combinations_count())
            
            row += 1
    
    def setup_metrics_section(self):
        """Setup metrics selection."""
        metrics_frame = ctk.CTkFrame(self.main_frame)
        metrics_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            metrics_frame,
            text="Optimization Metrics",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Metrics grid
        metrics_grid = ctk.CTkFrame(metrics_frame)
        metrics_grid.pack(fill="x", padx=10, pady=(0, 10))
        
        # Clustering metrics
        ctk.CTkLabel(
            metrics_grid,
            text="Clustering Metrics:",
            font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=5)
        
        clustering_metrics = [
            ("silhouette_score", "Silhouette Score", "Higher is better"),
            ("calinski_harabasz_score", "Calinski-Harabasz", "Higher is better"),
            ("davies_bouldin_score", "Davies-Bouldin", "Lower is better"),
        ]
        
        row = 1
        for metric_key, display_name, description in clustering_metrics:
            var = ctk.BooleanVar(value=metric_key == "silhouette_score")
            self.metric_vars[metric_key] = var
            
            check = ctk.CTkCheckBox(
                metrics_grid,
                text=display_name,
                variable=var
            )
            check.grid(row=row, column=0, padx=10, pady=2, sticky="w")
            
            ctk.CTkLabel(
                metrics_grid,
                text=description,
                font=ctk.CTkFont(size=11),
                text_color="gray"
            ).grid(row=row, column=1, padx=10, pady=2, sticky="w")
            
            row += 1
        
        # Other metrics
        ctk.CTkLabel(
            metrics_grid,
            text="Other Metrics:",
            font=ctk.CTkFont(weight="bold")
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(10, 5))
        row += 1
        
        other_metrics = [
            ("topic_diversity", "Topic Diversity", "Higher is better"),
            ("training_time", "Training Time", "Lower is better"),
        ]
        
        for metric_key, display_name, description in other_metrics:
            var = ctk.BooleanVar(value=False)
            self.metric_vars[metric_key] = var
            
            check = ctk.CTkCheckBox(
                metrics_grid,
                text=display_name,
                variable=var
            )
            check.grid(row=row, column=0, padx=10, pady=2, sticky="w")
            
            ctk.CTkLabel(
                metrics_grid,
                text=description,
                font=ctk.CTkFont(size=11),
                text_color="gray"
            ).grid(row=row, column=1, padx=10, pady=2, sticky="w")
            
            row += 1
        
        # Primary metric selection
        primary_frame = ctk.CTkFrame(metrics_frame)
        primary_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(primary_frame, text="Primary Metric:").pack(side="left", padx=(0, 10))
        
        metric_values = [key for key in self.metric_vars.keys()]
        self.primary_metric_menu = ctk.CTkOptionMenu(
            primary_frame,
            variable=self.primary_metric_var,
            values=metric_values,
            command=self._on_primary_metric_changed
        )
        self.primary_metric_menu.pack(side="left", padx=(0, 10))
        
        self.minimize_check = ctk.CTkCheckBox(
            primary_frame,
            text="Minimize",
            variable=self.minimize_var
        )
        self.minimize_check.pack(side="left")
    
    def setup_settings_section(self):
        """Setup optimization settings."""
        settings_frame = ctk.CTkFrame(self.main_frame)
        settings_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            settings_frame,
            text="Optimization Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Settings grid
        settings_grid = ctk.CTkFrame(settings_frame)
        settings_grid.pack(fill="x", padx=10, pady=(0, 10))
        
        # Parallel jobs
        ctk.CTkLabel(settings_grid, text="Parallel Jobs:").grid(
            row=0, column=0, padx=10, pady=5, sticky="w"
        )
        self.n_jobs_slider = ctk.CTkSlider(
            settings_grid,
            from_=1,
            to=8,
            number_of_steps=7,
            variable=self.n_jobs_var,
            command=lambda v: self.n_jobs_label.configure(text=str(int(v)))
        )
        self.n_jobs_slider.grid(row=0, column=1, padx=10, pady=5)
        self.n_jobs_label = ctk.CTkLabel(settings_grid, text="1")
        self.n_jobs_label.grid(row=0, column=2, padx=10, pady=5)
        
        # Cross-validation folds
        ctk.CTkLabel(settings_grid, text="CV Folds:").grid(
            row=1, column=0, padx=10, pady=5, sticky="w"
        )
        self.cv_folds_slider = ctk.CTkSlider(
            settings_grid,
            from_=2,
            to=10,
            number_of_steps=8,
            variable=self.cv_folds_var,
            command=lambda v: self.cv_folds_label.configure(text=str(int(v)))
        )
        self.cv_folds_slider.grid(row=1, column=1, padx=10, pady=5)
        self.cv_folds_label = ctk.CTkLabel(settings_grid, text="3")
        self.cv_folds_label.grid(row=1, column=2, padx=10, pady=5)
        
        # Early stopping
        early_stop_frame = ctk.CTkFrame(settings_grid)
        early_stop_frame.grid(row=2, column=0, columnspan=3, pady=10, sticky="w")
        
        self.early_stop_check = ctk.CTkCheckBox(
            early_stop_frame,
            text="Early Stopping",
            variable=self.early_stopping_var,
            command=self._on_early_stopping_changed
        )
        self.early_stop_check.pack(side="left", padx=10)
        
        ctk.CTkLabel(early_stop_frame, text="Patience:").pack(side="left", padx=(20, 5))
        self.patience_entry = ctk.CTkEntry(early_stop_frame, width=50)
        self.patience_entry.insert(0, "10")
        self.patience_entry.pack(side="left")
    
    def setup_control_section(self):
        """Setup control buttons."""
        control_frame = ctk.CTkFrame(self.main_frame)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        buttons_frame = ctk.CTkFrame(control_frame)
        buttons_frame.pack(pady=10)
        
        # Start button
        self.start_btn = ctk.CTkButton(
            buttons_frame,
            text="Start Optimization",
            command=self._start_optimization,
            width=150,
            fg_color="green"
        )
        self.start_btn.pack(side="left", padx=5)
        
        # Stop button
        self.stop_btn = ctk.CTkButton(
            buttons_frame,
            text="Stop",
            command=self._stop_optimization,
            width=100,
            state="disabled",
            fg_color="red"
        )
        self.stop_btn.pack(side="left", padx=5)
        
        # Apply best button
        self.apply_btn = ctk.CTkButton(
            buttons_frame,
            text="Apply Best Parameters",
            command=self._apply_best_parameters,
            width=150,
            state="disabled"
        )
        self.apply_btn.pack(side="left", padx=5)
        
        # Export button
        self.export_btn = ctk.CTkButton(
            buttons_frame,
            text="Export Results",
            command=self._export_results,
            width=100,
            state="disabled"
        )
        self.export_btn.pack(side="left", padx=5)
        
        # Progress bar
        self.progress_frame = ctk.CTkFrame(control_frame)
        self.progress_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.pack(fill="x", pady=5)
        self.progress_bar.set(0)
        
        self.progress_label = ctk.CTkLabel(
            self.progress_frame,
            text="Ready to start optimization"
        )
        self.progress_label.pack()
    
    def setup_results_section(self):
        """Setup results display section."""
        results_frame = ctk.CTkFrame(self.main_frame)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        ctk.CTkLabel(
            results_frame,
            text="Optimization Results",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Results notebook
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Summary tab
        self.summary_frame = ctk.CTkFrame(self.results_notebook)
        self.results_notebook.add(self.summary_frame, text="Summary")
        
        self.summary_text = ctk.CTkTextbox(self.summary_frame, height=200)
        self.summary_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Best runs tab
        self.best_runs_frame = ctk.CTkFrame(self.results_notebook)
        self.results_notebook.add(self.best_runs_frame, text="Best Runs")
        
        # Create treeview for best runs
        columns = ("Rank", "Score", "Parameters", "Time")
        self.best_runs_tree = ttk.Treeview(
            self.best_runs_frame,
            columns=columns,
            show="tree headings",
            height=10
        )
        
        for col in columns:
            self.best_runs_tree.heading(col, text=col)
            self.best_runs_tree.column(col, width=100)
        
        self.best_runs_tree.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Parameter importance tab
        self.importance_frame = ctk.CTkFrame(self.results_notebook)
        self.results_notebook.add(self.importance_frame, text="Parameter Importance")
        
        self.importance_text = ctk.CTkTextbox(self.importance_frame, height=200)
        self.importance_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def _on_frame_configure(self, event=None):
        """Update scroll region when frame size changes."""
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
    
    def _on_canvas_configure(self, event):
        """Update frame width when canvas size changes."""
        canvas_width = event.width
        self.main_canvas.itemconfig(self.main_canvas_frame, width=canvas_width)
    
    def _on_strategy_changed(self):
        """Handle strategy selection change."""
        strategy = self.strategy_var.get()
        
        if strategy == "grid_search":
            desc = "Grid Search: Exhaustive search through all parameter combinations"
        elif strategy == "random_search":
            desc = "Random Search: Sample random parameter combinations"
        elif strategy == "bayesian":
            desc = "Bayesian Optimization: Smart search using probabilistic model"
        else:
            desc = ""
        
        self.strategy_desc.configure(text=desc)
        self._update_combinations_count()
    
    def _on_primary_metric_changed(self, value):
        """Handle primary metric change."""
        # Update minimize checkbox based on metric
        if value in ["davies_bouldin_score", "training_time"]:
            self.minimize_var.set(True)
        else:
            self.minimize_var.set(False)
    
    def _on_early_stopping_changed(self):
        """Handle early stopping toggle."""
        if self.early_stopping_var.get():
            self.patience_entry.configure(state="normal")
        else:
            self.patience_entry.configure(state="disabled")
    
    def _update_combinations_count(self):
        """Update total combinations count for the search space."""
        try:
            strategy = self.strategy_var.get()
            
            # Calculate total combinations for enabled parameters
            total = 1
            invalid_params = []
            enabled_count = 0
            
            for param_name, widgets in self.param_widgets.items():
                if widgets["enabled"].get():
                    enabled_count += 1
                    try:
                        min_val = float(widgets["min"].get())
                        max_val = float(widgets["max"].get())
                        step_val = float(widgets["step"].get())
                        
                        # Validation
                        if min_val >= max_val:
                            invalid_params.append(f"{param_name}: min >= max")
                            continue
                        if step_val <= 0:
                            invalid_params.append(f"{param_name}: step <= 0")
                            continue
                        
                        # Calculate combinations for this parameter
                        if widgets["type"] == "int":
                            if step_val != int(step_val):
                                invalid_params.append(f"{param_name}: non-integer step for integer parameter")
                                continue
                            count = len(range(int(min_val), int(max_val) + 1, int(step_val)))
                        else:
                            count = int((max_val - min_val) / step_val) + 1
                        
                        if count <= 0:
                            invalid_params.append(f"{param_name}: no valid values")
                            continue
                            
                        total *= count
                        
                    except ValueError as e:
                        invalid_params.append(f"{param_name}: invalid number format")
                    except Exception as e:
                        invalid_params.append(f"{param_name}: {str(e)}")
            
            # Display results based on strategy and validation
            if invalid_params:
                self.combinations_label.configure(
                    text=f"❌ Invalid parameters: {'; '.join(invalid_params[:2])}{'...' if len(invalid_params) > 2 else ''}",
                    text_color="red"
                )
            elif enabled_count == 0:
                self.combinations_label.configure(
                    text="❌ No parameters enabled for optimization",
                    text_color="red"
                )
            else:
                if strategy == "grid_search":
                    if total > 1000000:  # 1 million combinations warning
                        self.combinations_label.configure(
                            text=f"⚠️  Grid search: {total:,} combinations (very large!)",
                            text_color="orange"
                        )
                    else:
                        self.combinations_label.configure(
                            text=f"✅ Grid search: {total:,} combinations",
                            text_color="green"
                        )
                elif strategy == "random_search":
                    self.combinations_label.configure(
                        text=f"✅ Random search: {total:,} possible combinations (will sample subset)",
                        text_color="green"
                    )
                elif strategy == "bayesian":
                    self.combinations_label.configure(
                        text=f"✅ Bayesian optimization: {total:,} possible combinations (intelligent sampling)",
                        text_color="green"
                    )
                else:
                    self.combinations_label.configure(
                        text=f"✅ Search space: {total:,} combinations",
                        text_color="green"
                    )
                    
        except Exception as e:
            self.combinations_label.configure(
                text=f"❌ Error calculating combinations: {str(e)}",
                text_color="red"
            )
    
    def _start_optimization(self):
        """Start hyperparameter optimization."""
        if not self.controller:
            messagebox.showerror("Error", "Optimization controller not initialized")
            return
        
        if not MODELS_AVAILABLE:
            messagebox.showerror("Error", "Optimization models not available")
            return
        
        # Build optimization config
        try:
            config = self._build_optimization_config()
            if not config:
                return
        except Exception as e:
            messagebox.showerror("Configuration Error", f"Failed to build configuration: {str(e)}")
            return
        
        # Create callbacks
        callbacks = OptimizationCallbacks(
            on_progress=self._on_progress,
            on_run_complete=self._on_run_complete,
            on_optimization_complete=self._on_optimization_complete,
            on_error=self._on_error
        )
        
        # Start optimization
        if self.controller.start_optimization(config, callbacks):
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.apply_btn.configure(state="disabled")
            self.export_btn.configure(state="disabled")
            
            # Clear previous results
            self.summary_text.delete("1.0", "end")
            self.best_runs_tree.delete(*self.best_runs_tree.get_children())
            self.importance_text.delete("1.0", "end")
            
            logger.info("Hyperparameter optimization started")
        else:
            messagebox.showerror("Error", "Failed to start optimization")
    
    def _build_optimization_config(self):
        """Build optimization configuration from UI settings."""
        # Get strategy
        strategy_map = {
            "grid_search": OptimizationStrategy.GRID_SEARCH,
            "random_search": OptimizationStrategy.RANDOM_SEARCH,
            "bayesian": OptimizationStrategy.BAYESIAN
        }
        
        strategy = strategy_map.get(self.strategy_var.get())
        if not strategy:
            messagebox.showerror("Error", "Invalid optimization strategy selected")
            return None
        
        # Build parameter space
        param_space = self._get_parameter_space()
        if not param_space.get_all_parameters():
            messagebox.showerror("Error", "No parameters selected for optimization")
            return None
        
        # Get selected metrics
        metrics = self._get_selected_metrics()
        if not metrics:
            messagebox.showerror("Error", "No metrics selected for optimization")
            return None
        
        # Get primary metric
        primary_metric_map = {
            "silhouette_score": MetricType.SILHOUETTE,
            "calinski_harabasz_score": MetricType.CALINSKI_HARABASZ,
            "davies_bouldin_score": MetricType.DAVIES_BOULDIN,
            "topic_diversity": MetricType.TOPIC_DIVERSITY,
            "training_time": MetricType.TRAINING_TIME
        }
        
        primary_metric = primary_metric_map.get(self.primary_metric_var.get())
        if not primary_metric:
            messagebox.showerror("Error", "Invalid primary metric selected")
            return None
        
        if primary_metric not in metrics:
            messagebox.showerror("Error", "Primary metric must be selected in the metrics list")
            return None
        
        # Build config
        config = OptimizationConfig(
            strategy=strategy,
            parameter_space=param_space,
            metrics=metrics,
            primary_metric=primary_metric,
            minimize_primary=self.minimize_var.get(),
            n_jobs=self.n_jobs_var.get(),
            cv_folds=self.cv_folds_var.get(),
            early_stopping=self.early_stopping_var.get(),
            patience=int(self.patience_entry.get()) if self.early_stopping_var.get() else 10,
            max_iterations=50 if strategy == OptimizationStrategy.RANDOM_SEARCH else None
        )
        
        return config
    
    def _get_selected_metrics(self) -> List[MetricType]:
        """Get list of selected metrics from UI."""
        selected = []
        
        metric_map = {
            "silhouette_score": MetricType.SILHOUETTE,
            "calinski_harabasz_score": MetricType.CALINSKI_HARABASZ,
            "davies_bouldin_score": MetricType.DAVIES_BOULDIN,
            "topic_diversity": MetricType.TOPIC_DIVERSITY,
            "training_time": MetricType.TRAINING_TIME
        }
        
        for metric_key, metric_var in self.metric_vars.items():
            if metric_var.get():
                if metric_key in metric_map:
                    selected.append(metric_map[metric_key])
        
        return selected
    
    def _get_parameter_space(self) -> ParameterSpace:
        """Build parameter space from UI settings."""
        param_space = ParameterSpace()
        
        for param_name, widgets in self.param_widgets.items():
            if widgets["enabled"].get():
                try:
                    min_val = float(widgets["min"].get())
                    max_val = float(widgets["max"].get())
                    step_val = float(widgets["step"].get())
                    
                    if widgets["type"] == "int":
                        param_range = ParameterRange(
                            name=param_name,
                            param_type="int",
                            min_value=int(min_val),
                            max_value=int(max_val),
                            step=int(step_val)
                        )
                    else:  # float
                        param_range = ParameterRange(
                            name=param_name,
                            param_type="float",
                            min_value=min_val,
                            max_value=max_val,
                            step=step_val
                        )
                    
                    # Set the parameter directly on the param_space object
                    setattr(param_space, param_name, param_range)
                    
                except ValueError as e:
                    logger.warning(f"Invalid parameter range for {param_name}: {e}")
        
        return param_space
    
    def _stop_optimization(self):
        """Stop the current optimization."""
        if self.controller:
            self.controller.stop_optimization()
            self.stop_btn.configure(state="disabled")
            self.progress_label.configure(text="Stopping optimization...")
    
    def _apply_best_parameters(self):
        """Apply the best parameters to topic modeling."""
        messagebox.showinfo("Apply Parameters", "Best parameters would be applied to topic modeling configuration")
    
    def _export_results(self):
        """Export optimization results."""
        messagebox.showinfo("Export Results", "Optimization results would be exported to file")
    
    def _on_progress(self, current: int, total: int, message: str):
        """Handle progress updates."""
        progress = current / total if total > 0 else 0
        self.progress_bar.set(progress)
        self.progress_label.configure(text=f"{message} ({current}/{total})")
    
    def _on_run_complete(self, run):
        """Handle individual run completion."""
        # Could update a live view of results here
        pass
    
    def _on_optimization_complete(self, result):
        """Handle optimization completion."""
        self.current_result = result
        
        # Update UI
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.apply_btn.configure(state="normal")
        self.export_btn.configure(state="normal")
        
        # Update progress
        self.progress_bar.set(1.0)
        self.progress_label.configure(
            text=f"Optimization complete: {result.completed_iterations} runs"
        )
        
        # Update results display
        self._display_results(result)
    
    def _on_error(self, error_msg: str):
        """Handle optimization error."""
        messagebox.showerror("Optimization Error", error_msg)
        
        # Reset UI
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.progress_label.configure(text="Optimization failed")
    
    def _display_results(self, result):
        """Display optimization results."""
        # Placeholder for results display
        summary = f"Optimization Summary\n"
        summary += f"{'=' * 50}\n\n"
        summary += f"Strategy: {result.config.strategy.value}\n"
        summary += f"Total runs: {result.total_iterations}\n"
        summary += f"Completed runs: {result.completed_iterations}\n"
        
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", summary) 