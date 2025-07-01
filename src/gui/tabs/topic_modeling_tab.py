"""
Topic Modeling Tab - Main BERTopic configuration and execution.
"""

import logging
import customtkinter as ctk
from typing import Optional

# Import tooltip utility
try:
    from ..utils.tooltip import add_tooltip, PARAMETER_DESCRIPTIONS
    TOOLTIPS_AVAILABLE = True
except ImportError:
    TOOLTIPS_AVAILABLE = False

from ...controllers import TopicModelingController
from ...models import (
    TopicModelConfig,
    TopicResult,
    TopicModelingProgress,
    ClusteringConfig,
    VectorizationConfig,
    UMAPConfig,
    RepresentationConfig,
)

logger = logging.getLogger(__name__)


class TopicModelingTab:
    """Tab for running topic modeling operations."""

    def __init__(self, parent):
        self.parent = parent
        self.controller: Optional[TopicModelingController] = None
        self.current_config: Optional[TopicModelConfig] = None
        self.current_result: Optional[TopicResult] = None

        self.setup_ui()
        logger.info("Topic Modeling tab initialized")

    def set_controller(self, controller: TopicModelingController):
        """Set the topic modeling controller."""
        self.controller = controller
        self.controller.set_callbacks(
            progress_callback=self.on_progress_update,
            completion_callback=self.on_training_complete,
            error_callback=self.on_error,
            status_callback=self.on_status_update,
        )

    def setup_ui(self):
        """Set up the tab user interface."""
        # Main content frame with scrollable area
        self.main_frame = ctk.CTkScrollableFrame(self.parent)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Topic Modeling Configuration",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.title_label.pack(pady=(10, 20))

        # Status section
        self.setup_status_section()

        # Configuration sections
        self.setup_clustering_section()
        self.setup_vectorization_section()
        self.setup_umap_section()
        self.setup_representation_section()
        self.setup_advanced_section()

        # Control buttons
        self.setup_control_buttons()

        # Results section
        self.setup_results_section()

        # Initialize with default configuration
        self.load_default_configuration()

    def setup_status_section(self):
        """Setup status display section."""
        status_frame = ctk.CTkFrame(self.main_frame)
        status_frame.pack(fill="x", pady=(0, 20))

        ctk.CTkLabel(status_frame, text="Status", font=ctk.CTkFont(weight="bold")).pack(
            pady=5
        )

        self.status_label = ctk.CTkLabel(status_frame, text="Ready for topic modeling")
        self.status_label.pack(pady=5)

        self.progress_bar = ctk.CTkProgressBar(status_frame)
        self.progress_bar.pack(fill="x", padx=20, pady=5)
        self.progress_bar.set(0)

    def setup_clustering_section(self):
        """Setup clustering configuration section."""
        cluster_frame = ctk.CTkFrame(self.main_frame)
        cluster_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            cluster_frame, text="Clustering Algorithm", font=ctk.CTkFont(weight="bold")
        ).pack(pady=5)

        # Algorithm selection
        alg_frame = ctk.CTkFrame(cluster_frame)
        alg_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(alg_frame, text="Algorithm:").pack(side="left", padx=5)
        self.clustering_algorithm = ctk.CTkOptionMenu(
            alg_frame,
            values=["hdbscan", "kmeans", "agglomerative", "optics"],
            command=self.on_clustering_algorithm_change,
        )
        self.clustering_algorithm.pack(side="left", padx=5)
        self.clustering_algorithm.set("hdbscan")

        # Parameters frame (will be populated based on algorithm)
        self.clustering_params_frame = ctk.CTkFrame(cluster_frame)
        self.clustering_params_frame.pack(fill="x", padx=10, pady=5)

        self.setup_clustering_params()

    def setup_clustering_params(self):
        """Setup clustering parameters based on selected algorithm."""
        # Clear existing params
        for widget in self.clustering_params_frame.winfo_children():
            widget.destroy()

        algorithm = self.clustering_algorithm.get()

        if algorithm == "hdbscan":
            self.setup_hdbscan_params()
        elif algorithm == "kmeans":
            self.setup_kmeans_params()
        elif algorithm == "agglomerative":
            self.setup_agglomerative_params()

    def setup_hdbscan_params(self):
        """Setup HDBSCAN parameters."""
        # Min cluster size
        size_frame = ctk.CTkFrame(self.clustering_params_frame)
        size_frame.pack(fill="x", pady=2)
        size_label = ctk.CTkLabel(size_frame, text="Min Cluster Size:")
        size_label.pack(side="left", padx=5)
        if TOOLTIPS_AVAILABLE:
            add_tooltip(size_label, PARAMETER_DESCRIPTIONS.get("min_cluster_size", "Minimum number of documents required to form a cluster"))
        
        self.hdbscan_min_cluster_size = ctk.CTkEntry(size_frame, width=100)
        self.hdbscan_min_cluster_size.pack(side="left", padx=5)
        self.hdbscan_min_cluster_size.insert(0, "10")

        # Metric
        metric_frame = ctk.CTkFrame(self.clustering_params_frame)
        metric_frame.pack(fill="x", pady=2)
        metric_label = ctk.CTkLabel(metric_frame, text="Metric:")
        metric_label.pack(side="left", padx=5)
        if TOOLTIPS_AVAILABLE:
            add_tooltip(metric_label, PARAMETER_DESCRIPTIONS.get("metric", "Distance metric for measuring similarity between documents"))
        self.hdbscan_metric = ctk.CTkOptionMenu(
            metric_frame, values=["euclidean", "cosine", "manhattan"]
        )
        self.hdbscan_metric.pack(side="left", padx=5)
        self.hdbscan_metric.set("euclidean")

    def setup_kmeans_params(self):
        """Setup K-Means parameters."""
        # Number of clusters
        k_frame = ctk.CTkFrame(self.clustering_params_frame)
        k_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(k_frame, text="Number of Clusters:").pack(side="left", padx=5)
        self.kmeans_n_clusters = ctk.CTkEntry(k_frame, width=100)
        self.kmeans_n_clusters.pack(side="left", padx=5)
        self.kmeans_n_clusters.insert(0, "8")

    def setup_agglomerative_params(self):
        """Setup Agglomerative clustering parameters."""
        # Number of clusters
        k_frame = ctk.CTkFrame(self.clustering_params_frame)
        k_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(k_frame, text="Number of Clusters:").pack(side="left", padx=5)
        self.agg_n_clusters = ctk.CTkEntry(k_frame, width=100)
        self.agg_n_clusters.pack(side="left", padx=5)
        self.agg_n_clusters.insert(0, "8")

        # Linkage
        linkage_frame = ctk.CTkFrame(self.clustering_params_frame)
        linkage_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(linkage_frame, text="Linkage:").pack(side="left", padx=5)
        self.agg_linkage = ctk.CTkOptionMenu(
            linkage_frame, values=["ward", "complete", "average", "single"]
        )
        self.agg_linkage.pack(side="left", padx=5)
        self.agg_linkage.set("ward")

    def setup_vectorization_section(self):
        """Setup vectorization configuration section."""
        vec_frame = ctk.CTkFrame(self.main_frame)
        vec_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            vec_frame, text="Vectorization", font=ctk.CTkFont(weight="bold")
        ).pack(pady=5)

        # Vectorizer type
        type_frame = ctk.CTkFrame(vec_frame)
        type_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(type_frame, text="Type:").pack(side="left", padx=5)
        self.vectorizer_type = ctk.CTkOptionMenu(type_frame, values=["tfidf", "count"])
        self.vectorizer_type.pack(side="left", padx=5)
        self.vectorizer_type.set("tfidf")

        # N-gram range
        ngram_frame = ctk.CTkFrame(vec_frame)
        ngram_frame.pack(fill="x", padx=10, pady=5)

        ngram_label = ctk.CTkLabel(ngram_frame, text="N-gram Range:")
        ngram_label.pack(side="left", padx=5)
        if TOOLTIPS_AVAILABLE:
            add_tooltip(ngram_label, PARAMETER_DESCRIPTIONS.get("ngram_range", "Range of n-grams to extract (e.g., 1-1 for single words, 1-2 for words and bigrams)"))
        self.ngram_min = ctk.CTkEntry(ngram_frame, width=50)
        self.ngram_min.pack(side="left", padx=2)
        self.ngram_min.insert(0, "1")

        ctk.CTkLabel(ngram_frame, text="to").pack(side="left", padx=2)
        self.ngram_max = ctk.CTkEntry(ngram_frame, width=50)
        self.ngram_max.pack(side="left", padx=2)
        self.ngram_max.insert(0, "1")

        # Stop words
        stop_frame = ctk.CTkFrame(vec_frame)
        stop_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(stop_frame, text="Stop Words:").pack(side="left", padx=5)
        self.stop_words = ctk.CTkOptionMenu(stop_frame, values=["english", "none"])
        self.stop_words.pack(side="left", padx=5)
        self.stop_words.set("english")

    def setup_umap_section(self):
        """Setup UMAP configuration section."""
        umap_frame = ctk.CTkFrame(self.main_frame)
        umap_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            umap_frame, text="UMAP Configuration", font=ctk.CTkFont(weight="bold")
        ).pack(pady=5)

        # Parameters in a grid-like layout
        params_frame = ctk.CTkFrame(umap_frame)
        params_frame.pack(fill="x", padx=10, pady=5)

        # N neighbors
        neigh_frame = ctk.CTkFrame(params_frame)
        neigh_frame.pack(fill="x", pady=2)
        neigh_label = ctk.CTkLabel(neigh_frame, text="N Neighbors:")
        neigh_label.pack(side="left", padx=5)
        if TOOLTIPS_AVAILABLE:
            add_tooltip(neigh_label, PARAMETER_DESCRIPTIONS.get("n_neighbors", "Number of neighbors for UMAP dimensionality reduction"))
        
        self.umap_n_neighbors = ctk.CTkEntry(neigh_frame, width=100)
        self.umap_n_neighbors.pack(side="left", padx=5)
        self.umap_n_neighbors.insert(0, "15")

        # Min distance
        dist_frame = ctk.CTkFrame(params_frame)
        dist_frame.pack(fill="x", pady=2)
        dist_label = ctk.CTkLabel(dist_frame, text="Min Distance:")
        dist_label.pack(side="left", padx=5)
        if TOOLTIPS_AVAILABLE:
            add_tooltip(dist_label, PARAMETER_DESCRIPTIONS.get("min_dist", "Minimum distance between points in UMAP embedding"))
        
        self.umap_min_dist = ctk.CTkEntry(dist_frame, width=100)
        self.umap_min_dist.pack(side="left", padx=5)
        self.umap_min_dist.insert(0, "0.1")

        # Metric
        metric_frame = ctk.CTkFrame(params_frame)
        metric_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(metric_frame, text="Metric:").pack(side="left", padx=5)
        self.umap_metric = ctk.CTkOptionMenu(
            metric_frame, values=["cosine", "euclidean", "manhattan"]
        )
        self.umap_metric.pack(side="left", padx=5)
        self.umap_metric.set("cosine")

    def setup_representation_section(self):
        """Setup representation models section."""
        rep_frame = ctk.CTkFrame(self.main_frame)
        rep_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            rep_frame, text="Representation Models", font=ctk.CTkFont(weight="bold")
        ).pack(pady=5)

        # Enable representation models
        self.use_representation = ctk.CTkCheckBox(
            rep_frame, text="Use Representation Models"
        )
        self.use_representation.pack(pady=5)
        if TOOLTIPS_AVAILABLE:
            add_tooltip(self.use_representation, PARAMETER_DESCRIPTIONS.get("use_representation", "Enable advanced language models to improve topic descriptions"))
        self.use_representation.select()  # Enabled by default with fixed architecture

        # Model options
        models_frame = ctk.CTkFrame(rep_frame)
        models_frame.pack(fill="x", padx=10, pady=5)

        self.keybert_var = ctk.CTkCheckBox(models_frame, text="KeyBERT")
        self.keybert_var.pack(side="left", padx=10)
        self.keybert_var.select()  # Enabled by default

        self.mmr_var = ctk.CTkCheckBox(models_frame, text="MaximalMarginalRelevance")
        self.mmr_var.pack(side="left", padx=10)

    def setup_advanced_section(self):
        """Setup advanced options section."""
        adv_frame = ctk.CTkFrame(self.main_frame)
        adv_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            adv_frame, text="Advanced Options", font=ctk.CTkFont(weight="bold")
        ).pack(pady=5)

        # Top K words
        topk_frame = ctk.CTkFrame(adv_frame)
        topk_frame.pack(fill="x", padx=10, pady=5)

        topk_label = ctk.CTkLabel(topk_frame, text="Top K Words:")
        topk_label.pack(side="left", padx=5)
        if TOOLTIPS_AVAILABLE:
            add_tooltip(topk_label, PARAMETER_DESCRIPTIONS.get("top_n_words", "Number of most representative words to extract for each topic"))
        
        self.top_k_words = ctk.CTkEntry(topk_frame, width=100)
        self.top_k_words.pack(side="left", padx=5)
        self.top_k_words.insert(0, "10")

        # Calculate probabilities
        self.calc_probabilities = ctk.CTkCheckBox(
            adv_frame, text="Calculate Topic Probabilities"
        )
        self.calc_probabilities.pack(pady=5)

        # Save model
        self.save_model = ctk.CTkCheckBox(adv_frame, text="Save Trained Model")
        self.save_model.pack(pady=5)
        self.save_model.select()

    def setup_control_buttons(self):
        """Setup control buttons."""
        control_frame = ctk.CTkFrame(self.main_frame)
        control_frame.pack(fill="x", pady=(0, 20))

        self.start_button = ctk.CTkButton(
            control_frame,
            text="Start Topic Modeling",
            command=self.start_topic_modeling,
            font=ctk.CTkFont(weight="bold"),
        )
        self.start_button.pack(side="left", padx=10, pady=10)

        self.cancel_button = ctk.CTkButton(
            control_frame,
            text="Cancel",
            command=self.cancel_topic_modeling,
            state="disabled",
        )
        self.cancel_button.pack(side="left", padx=5, pady=10)

        self.clear_button = ctk.CTkButton(
            control_frame, text="Clear Results", command=self.clear_results
        )
        self.clear_button.pack(side="left", padx=5, pady=10)

    def setup_results_section(self):
        """Setup results display section."""
        results_frame = ctk.CTkFrame(self.main_frame)
        results_frame.pack(fill="both", expand=True, pady=(0, 10))

        ctk.CTkLabel(
            results_frame, text="Results", font=ctk.CTkFont(weight="bold")
        ).pack(pady=5)

        # Results text area
        self.results_text = ctk.CTkTextbox(results_frame, height=200)
        self.results_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.results_text.insert(
            "1.0", "No results yet. Configure parameters and start topic modeling."
        )

    def on_clustering_algorithm_change(self, value):
        """Handle clustering algorithm change."""
        self.setup_clustering_params()

    def load_default_configuration(self):
        """Load default configuration values."""
        if self.controller:
            self.current_config = self.controller.create_default_config()

    def get_current_configuration(self) -> TopicModelConfig:
        """Get configuration from UI elements."""
        try:
            # Clustering config
            clustering_config = ClusteringConfig(
                algorithm=self.clustering_algorithm.get()
            )

            if clustering_config.algorithm == "hdbscan":
                clustering_config.min_cluster_size = int(
                    self.hdbscan_min_cluster_size.get() or "10"
                )
                clustering_config.metric = self.hdbscan_metric.get()
            elif clustering_config.algorithm == "kmeans":
                clustering_config.n_clusters = int(self.kmeans_n_clusters.get() or "8")
            elif clustering_config.algorithm == "agglomerative":
                clustering_config.n_clusters = int(self.agg_n_clusters.get() or "8")
                clustering_config.linkage = self.agg_linkage.get()

            # Vectorization config
            vectorization_config = VectorizationConfig(
                vectorizer_type=self.vectorizer_type.get(),
                ngram_range=(
                    int(self.ngram_min.get() or "1"),
                    int(self.ngram_max.get() or "1"),
                ),
                stop_words=self.stop_words.get()
                if self.stop_words.get() != "none"
                else None,
            )

            # UMAP config
            umap_config = UMAPConfig(
                n_neighbors=int(self.umap_n_neighbors.get() or "15"),
                min_dist=float(self.umap_min_dist.get() or "0.1"),
                metric=self.umap_metric.get(),
            )

            # Representation config
            representation_models = []
            if self.use_representation.get():
                if self.keybert_var.get():
                    representation_models.append("KeyBERT")
                if self.mmr_var.get():
                    representation_models.append("MaximalMarginalRelevance")

            representation_config = RepresentationConfig(
                use_representation=self.use_representation.get(),
                representation_models=representation_models,
            )

            # Create full config
            config = TopicModelConfig(
                clustering_config=clustering_config,
                vectorization_config=vectorization_config,
                umap_config=umap_config,
                representation_config=representation_config,
                top_k_words=int(self.top_k_words.get() or "10"),
                calculate_probabilities=self.calc_probabilities.get(),
                save_model=self.save_model.get(),
            )

            return config

        except Exception as e:
            logger.error(f"Failed to get configuration: {e}")
            raise ValueError(f"Invalid configuration: {str(e)}")

    def start_topic_modeling(self):
        """Start topic modeling process."""
        if not self.controller:
            self.show_error("No controller available")
            return

        try:
            # Get configuration
            config = self.get_current_configuration()

            # Get data and embeddings from main window
            main_window = self._get_main_window()
            if not main_window:
                self.show_error("Cannot access main window")
                return

            # Get data configuration
            data_config = main_window.get_data_config()
            if not data_config or not data_config.is_configured:
                self.show_error(
                    "Please load and configure data in the Data Import tab first"
                )
                return

            # Get embedding result
            embedding_result = main_window.get_embedding_result()
            if not embedding_result or embedding_result.embeddings is None:
                self.show_error(
                    "Please generate embeddings in the Model Config tab first"
                )
                return

            # Set embedding config in topic config
            if hasattr(embedding_result, "model_info") and embedding_result.model_info:
                # Create EmbeddingConfig from ModelInfo
                from ...models.data_models import EmbeddingConfig

                embedding_config = EmbeddingConfig(
                    model_info=embedding_result.model_info,
                    batch_size=32,  # Default value
                    normalize_embeddings=True,  # Default value
                    device="cpu",  # Use CPU to avoid device issues
                )
                config.embedding_config = embedding_config
            else:
                self.show_error("No valid embedding model information available")
                return

            # Validate configuration
            is_valid, errors = self.controller.validate_configuration(config)
            if not is_valid:
                error_msg = "Configuration validation failed:\n" + "\n".join(errors)
                self.show_error(error_msg)
                return

            # Start topic modeling
            success = self.controller.start_topic_modeling(
                data_config=data_config,
                topic_config=config,
                embedding_result=embedding_result,
            )

            if not success:
                self.show_error("Failed to start topic modeling")

        except Exception as e:
            self.show_error(f"Failed to start topic modeling: {str(e)}")

    def _get_main_window(self):
        """Get reference to main window."""
        # Navigate up the widget hierarchy to find main window
        widget = self.parent
        while widget:
            if hasattr(widget, "get_data_config"):
                return widget
            widget = widget.master
        return None

    def cancel_topic_modeling(self):
        """Cancel ongoing topic modeling."""
        if self.controller:
            self.controller.cancel_topic_modeling()

    def clear_results(self):
        """Clear current results."""
        if self.controller:
            self.controller.clear_current_results()
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", "Results cleared.")

    def on_progress_update(self, progress: TopicModelingProgress):
        """Handle progress updates."""
        self.progress_bar.set(progress.progress_percentage / 100.0)
        self.status_label.configure(text=progress.progress_text)

        if progress.stage == "complete":
            self.start_button.configure(state="normal")
            self.cancel_button.configure(state="disabled")
        elif progress.stage == "error":
            self.start_button.configure(state="normal")
            self.cancel_button.configure(state="disabled")
        else:
            self.start_button.configure(state="disabled")
            self.cancel_button.configure(state="normal")

    def on_training_complete(self, result: TopicResult):
        """Handle training completion."""
        self.current_result = result
        self.display_results(result)

    def on_error(self, error_message: str):
        """Handle errors."""
        self.show_error(error_message)

    def on_status_update(self, status: str):
        """Handle status updates."""
        self.status_label.configure(text=status)

    def display_results(self, result: TopicResult):
        """Display topic modeling results."""
        self.results_text.delete("1.0", "end")

        results_text = f"""Topic Modeling Results
========================

🎯 Topics Found: {result.num_topics}
📊 Total Documents: {result.num_documents}
⚠️  Outliers: {result.outlier_count} ({result.outlier_percentage:.1f}%)
⏱️  Training Time: {result.training_time_seconds:.2f} seconds

"""

        if result.silhouette_score:
            results_text += f"📈 Silhouette Score: {result.silhouette_score:.3f}\n"
        if result.calinski_harabasz_score:
            results_text += (
                f"📈 Calinski-Harabasz Score: {result.calinski_harabasz_score:.2f}\n"
            )
        if result.davies_bouldin_score:
            results_text += (
                f"📈 Davies-Bouldin Score: {result.davies_bouldin_score:.3f}\n"
            )

        results_text += "\n🏷️  Top Topics:\n" + "=" * 20 + "\n"

        # Show top topics
        for i, topic in enumerate(result.topic_info[:10]):  # Top 10 topics
            if topic.topic_id == -1:
                continue
            results_text += f"\nTopic {topic.topic_id} ({topic.size} docs, {topic.percentage:.1f}%):\n"
            results_text += f"  Words: {topic.top_words_string}\n"

        self.results_text.insert("1.0", results_text)

    def show_error(self, message: str):
        """Show error message."""
        self.status_label.configure(text=f"Error: {message}")
        logger.error(message)
