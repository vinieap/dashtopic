"""
Visualization Tab - Interactive plots and visualizations for topic modeling results.
"""

import logging
from tkinter import ttk
import customtkinter as ctk
from typing import Optional, List
import numpy as np

# Plotting libraries
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from wordcloud import WordCloud

from ...models import TopicResult
from ...controllers import TopicModelingController

logger = logging.getLogger(__name__)


class VisualizationTab:
    """Tab for displaying interactive visualizations of topic modeling results."""

    def __init__(self, parent):
        self.parent = parent
        self.controller: Optional[TopicModelingController] = None
        self.current_result: Optional[TopicResult] = None
        self.current_topic_selection: Optional[int] = None

        # Visualization state
        self.plot_type = ctk.StringVar(value="topic_distribution")
        self.color_scheme = ctk.StringVar(value="viridis")
        self.show_outliers = ctk.BooleanVar(value=False)
        self.num_documents = ctk.IntVar(value=100)  # Number of documents to use

        self.setup_ui()
        logger.info("Visualization tab initialized")

    def set_controller(self, controller: TopicModelingController):
        """Set the topic modeling controller."""
        self.controller = controller
        # Check if there are already results available
        self.refresh_visualizations()

    def setup_ui(self):
        """Set up the tab user interface."""
        # Main content frame
        self.main_frame = ctk.CTkFrame(self.parent)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Topic Visualizations",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.title_label.pack(pady=(10, 20))

        # Control panel
        self.setup_control_panel()

        # Visualization area
        self.setup_visualization_area()

        # Topic analysis panel
        self.setup_topic_analysis_panel()

        # Initialize with no data message
        self.show_no_data_message()

    def setup_control_panel(self):
        """Setup visualization control panel."""
        control_frame = ctk.CTkFrame(self.main_frame)
        control_frame.pack(fill="x", pady=(0, 10))

        # Control title
        ctk.CTkLabel(
            control_frame,
            text="Visualization Controls",
            font=ctk.CTkFont(weight="bold"),
        ).pack(pady=5)

        # Controls grid
        controls_grid = ctk.CTkFrame(control_frame)
        controls_grid.pack(fill="x", padx=10, pady=5)

        # Row 1: Plot type and refresh
        row1_frame = ctk.CTkFrame(controls_grid)
        row1_frame.pack(fill="x", pady=2)

        ctk.CTkLabel(row1_frame, text="Plot Type:").pack(side="left", padx=(10, 5))
        self.plot_type_menu = ctk.CTkOptionMenu(
            row1_frame,
            variable=self.plot_type,
            values=[
                "topic_distribution",
                "document_scatter_2d",
                "document_scatter_3d",
                "topic_heatmap",
                "word_cloud",
                "topic_evolution",
            ],
            command=self.on_plot_type_change,
        )
        self.plot_type_menu.pack(side="left", padx=5)

        # Refresh button
        self.refresh_btn = ctk.CTkButton(
            row1_frame, text="Refresh", command=self.refresh_visualizations, width=80
        )
        self.refresh_btn.pack(side="right", padx=(5, 10))

        # Row 2: Color scheme and options
        row2_frame = ctk.CTkFrame(controls_grid)
        row2_frame.pack(fill="x", pady=2)

        ctk.CTkLabel(row2_frame, text="Color Scheme:").pack(side="left", padx=(10, 5))
        self.color_menu = ctk.CTkOptionMenu(
            row2_frame,
            variable=self.color_scheme,
            values=["viridis", "plasma", "inferno", "magma", "tab10", "Set3"],
            command=self.on_color_change,
        )
        self.color_menu.pack(side="left", padx=5)

        # Options
        self.outliers_check = ctk.CTkCheckBox(
            row2_frame,
            text="Show Outliers",
            variable=self.show_outliers,
            command=self.on_option_change,
        )
        self.outliers_check.pack(side="left", padx=10)

        # Export button
        self.export_btn = ctk.CTkButton(
            row2_frame, text="Export Plot", command=self.export_current_plot, width=100
        )
        self.export_btn.pack(side="right", padx=(5, 10))

        # Row 3: Document count slider
        row3_frame = ctk.CTkFrame(controls_grid)
        row3_frame.pack(fill="x", pady=2)

        ctk.CTkLabel(row3_frame, text="Documents to Use:").pack(
            side="left", padx=(10, 5)
        )

        self.doc_count_label = ctk.CTkLabel(row3_frame, text="100")
        self.doc_count_label.pack(side="left", padx=5)

        self.doc_slider = ctk.CTkSlider(
            row3_frame,
            from_=10,
            to=1000,
            number_of_steps=99,
            variable=self.num_documents,
            command=self.on_doc_count_change,
            width=200,
        )
        self.doc_slider.pack(side="left", padx=10)

        # Update button for document count changes
        self.update_docs_btn = ctk.CTkButton(
            row3_frame,
            text="Update Data",
            command=self.update_document_count,
            width=100,
        )
        self.update_docs_btn.pack(side="left", padx=10)

    def setup_visualization_area(self):
        """Setup main visualization display area."""
        # Create notebook for different plot types
        self.viz_notebook = ttk.Notebook(self.main_frame)
        self.viz_notebook.pack(fill="both", expand=True, pady=(0, 10))

        # Matplotlib frame
        self.matplotlib_frame = ctk.CTkFrame(self.viz_notebook)
        self.viz_notebook.add(self.matplotlib_frame, text="Static Plots")

        # Plotly frame (for future interactive plots)
        self.plotly_frame = ctk.CTkFrame(self.viz_notebook)
        self.viz_notebook.add(self.plotly_frame, text="Interactive Plots")

        # Setup matplotlib canvas
        self.setup_matplotlib_canvas()

        # Setup plotly display
        self.setup_plotly_display()

    def setup_matplotlib_canvas(self):
        """Setup matplotlib canvas for static plots."""
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.fig.patch.set_facecolor("white")

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self.matplotlib_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        # Add navigation toolbar
        toolbar_frame = ctk.CTkFrame(self.matplotlib_frame)
        toolbar_frame.pack(fill="x", padx=5, pady=(0, 5))

        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

    def setup_plotly_display(self):
        """Setup plotly display area for interactive plots."""
        # Create controls for Plotly plots
        plotly_controls = ctk.CTkFrame(self.plotly_frame)
        plotly_controls.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            plotly_controls, text="Interactive Plot:", font=ctk.CTkFont(weight="bold")
        ).pack(side="left", padx=10)

        self.plotly_plot_type = ctk.StringVar(value="interactive_scatter")
        plotly_menu = ctk.CTkOptionMenu(
            plotly_controls,
            variable=self.plotly_plot_type,
            values=[
                "interactive_scatter",
                "interactive_distribution",
                "interactive_heatmap",
            ],
            command=self.on_plotly_plot_change,
            width=150,
        )
        plotly_menu.pack(side="left", padx=10)

        generate_btn = ctk.CTkButton(
            plotly_controls,
            text="Generate Interactive Plot",
            command=self.generate_plotly_visualization,
            width=160,
        )
        generate_btn.pack(side="left", padx=10)

        export_html_btn = ctk.CTkButton(
            plotly_controls,
            text="Export HTML",
            command=self.export_plotly_html,
            width=100,
        )
        export_html_btn.pack(side="right", padx=10)

        # Display area for Plotly plots with tkinterweb
        try:
            import tkinterweb

            self.plotly_display = tkinterweb.HtmlFrame(
                self.plotly_frame, messages_enabled=False
            )
            self.plotly_display.pack(fill="both", expand=True, padx=10, pady=(0, 10))
            self.plotly_display.load_html(
                "<html><body style='font-family: Arial; padding: 20px; text-align: center;'>"
                "<h3>Interactive Plotly Visualizations</h3>"
                "<p>Click 'Generate Interactive Plot' to create an interactive visualization.</p>"
                "<p>The plot will be displayed directly in this area with full interactivity.</p>"
                "</body></html>"
            )
            self.has_tkinterweb = True
        except ImportError:
            # Fallback to text widget if tkinterweb not available
            self.plotly_display = ctk.CTkTextbox(self.plotly_frame, height=400)
            self.plotly_display.pack(fill="both", expand=True, padx=10, pady=(0, 10))
            self.plotly_display.insert(
                "1.0",
                "tkinterweb not installed - showing HTML code instead.\n\nInstall tkinterweb for interactive plot display:\npip install tkinterweb\n\nClick 'Generate Interactive Plot' to create visualization code.",
            )
            self.has_tkinterweb = False

    def setup_topic_analysis_panel(self):
        """Setup topic analysis and exploration panel."""
        analysis_frame = ctk.CTkFrame(self.main_frame)
        analysis_frame.pack(fill="x", pady=(0, 10))

        # Analysis title
        ctk.CTkLabel(
            analysis_frame, text="Topic Analysis", font=ctk.CTkFont(weight="bold")
        ).pack(pady=5)

        # Topic selector and info
        selector_frame = ctk.CTkFrame(analysis_frame)
        selector_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(selector_frame, text="Select Topic:").pack(
            side="left", padx=(10, 5)
        )

        self.topic_var = ctk.StringVar(value="None")
        self.topic_selector = ctk.CTkOptionMenu(
            selector_frame,
            variable=self.topic_var,
            values=["None"],
            command=self.on_topic_selection_change,
            width=150,
        )
        self.topic_selector.pack(side="left", padx=5)

        # Topic info display
        self.topic_info_text = ctk.CTkTextbox(analysis_frame, height=100)
        self.topic_info_text.pack(fill="x", padx=10, pady=(0, 10))
        self.topic_info_text.insert(
            "1.0", "Select a topic to view detailed information..."
        )

    def show_no_data_message(self):
        """Show message when no data is available."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "No topic modeling results available.\n\nRun topic modeling in the Topic Modeling tab first.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        self.canvas.draw()

    def refresh_visualizations(self):
        """Refresh visualizations with current results."""
        if not self.controller:
            logger.warning("No controller available for visualization refresh")
            return

        # Get current results
        result = self.controller.get_current_result()
        if not result:
            logger.info("No topic modeling results available")
            self.show_no_data_message()
            self.update_topic_selector([])
            return

        self.current_result = result
        logger.info(f"Refreshing visualizations with {result.num_topics} topics")

        # Update topic selector
        topic_options = [
            f"Topic {topic.topic_id}: {topic.top_words_string[:30]}..."
            for topic in result.topic_info
            if topic.topic_id != -1
        ]
        if self.show_outliers.get() and any(
            t.topic_id == -1 for t in result.topic_info
        ):
            topic_options.append("Topic -1: Outliers")

        self.update_topic_selector(topic_options)

        # Generate visualization
        self.generate_visualization()

    def update_topic_selector(self, options: List[str]):
        """Update topic selector options."""
        if not options:
            options = ["None"]

        self.topic_selector.configure(values=options)
        if options[0] != "None":
            self.topic_var.set(options[0])
        else:
            self.topic_var.set("None")

    def generate_visualization(self):
        """Generate the selected visualization."""
        if not self.current_result:
            self.show_no_data_message()
            return

        plot_type = self.plot_type.get()
        logger.info(f"Generating {plot_type} visualization")

        try:
            if plot_type == "topic_distribution":
                self.plot_topic_distribution()
            elif plot_type == "document_scatter_2d":
                self.plot_document_scatter(is_3d=False)
            elif plot_type == "document_scatter_3d":
                self.plot_document_scatter(is_3d=True)
            elif plot_type == "topic_heatmap":
                self.plot_topic_heatmap()
            elif plot_type == "word_cloud":
                self.plot_word_cloud()
            elif plot_type == "topic_evolution":
                self.plot_topic_evolution()
            else:
                self.show_no_data_message()

        except Exception as e:
            logger.error(f"Failed to generate {plot_type} visualization: {e}")
            self.show_error_message(f"Failed to generate visualization: {str(e)}")

    def plot_topic_distribution(self):
        """Plot topic size distribution."""
        self.fig.clear()

        # Filter topics based on outlier setting
        topics_to_plot = [
            t
            for t in self.current_result.topic_info
            if t.topic_id != -1 or self.show_outliers.get()
        ]

        if not topics_to_plot:
            self.show_no_data_message()
            return

        # Prepare data
        topic_ids = [t.topic_id for t in topics_to_plot]
        topic_sizes = [t.size for t in topics_to_plot]
        topic_labels = [
            f"Topic {t.topic_id}" if t.topic_id != -1 else "Outliers"
            for t in topics_to_plot
        ]

        # Create bar plot
        ax = self.fig.add_subplot(111)
        colors = plt.cm.get_cmap(self.color_scheme.get())(
            np.linspace(0, 1, len(topic_ids))
        )

        bars = ax.bar(range(len(topic_ids)), topic_sizes, color=colors)

        # Customize plot
        ax.set_xlabel("Topics")
        ax.set_ylabel("Number of Documents")
        ax.set_title(
            f"Topic Distribution ({self.current_result.num_topics} topics, {self.current_result.num_documents} documents)"
        )
        ax.set_xticks(range(len(topic_ids)))
        ax.set_xticklabels(topic_labels, rotation=45, ha="right")

        # Add value labels on bars
        for bar, size in zip(bars, topic_sizes):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(topic_sizes) * 0.01,
                f"{size}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        self.fig.tight_layout()
        self.canvas.draw()

    def plot_document_scatter(self, is_3d=False):
        """Plot document embeddings in 2D/3D space."""
        if self.current_result.embeddings is None:
            self.show_error_message("No embeddings available for scatter plot")
            return

        self.fig.clear()

        # Generate appropriate UMAP embeddings for the plot type
        embeddings = self._get_umap_embeddings_for_plot(is_3d)
        if embeddings is None:
            return

        topics = np.array(self.current_result.topics)

        # Filter outliers if needed
        if not self.show_outliers.get():
            mask = topics != -1
            embeddings = embeddings[mask]
            topics = topics[mask]

        if len(embeddings) == 0:
            self.show_error_message("No data to plot after filtering")
            return

        # Create scatter plot
        if is_3d and embeddings.shape[1] >= 3:
            ax = self.fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                embeddings[:, 2],
                c=topics,
                cmap=self.color_scheme.get(),
                alpha=0.6,
                s=20,
            )
            ax.set_zlabel("UMAP 3")
            ax.set_title("Document Embeddings - 3D View (Colored by Topic)")
        else:
            ax = self.fig.add_subplot(111)
            scatter = ax.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                c=topics,
                cmap=self.color_scheme.get(),
                alpha=0.6,
                s=20,
            )
            ax.set_title("Document Embeddings - 2D View (Colored by Topic)")

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        # Add colorbar
        cbar = self.fig.colorbar(scatter, ax=ax)
        cbar.set_label("Topic ID")

        self.fig.tight_layout()
        self.canvas.draw()

    def _get_umap_embeddings_for_plot(self, is_3d=False):
        """Get UMAP embeddings for plotting, generating them if needed."""
        # Check if we already have the right dimensionality
        if self.current_result.umap_embeddings is not None:
            current_dims = self.current_result.umap_embeddings.shape[1]
            required_dims = 3 if is_3d else 2

            if current_dims >= required_dims:
                return self.current_result.umap_embeddings

        # Generate new UMAP embeddings with the required dimensionality
        try:
            from umap import UMAP

            n_components = 3 if is_3d else 2
            umap_model = UMAP(
                n_components=n_components,
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )

            logger.info(f"Generating {n_components}D UMAP embeddings for visualization")
            embeddings = umap_model.fit_transform(self.current_result.embeddings)

            # Cache the embeddings in the result if it's the default 2D
            if not is_3d:
                self.current_result.umap_embeddings = embeddings

            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate UMAP embeddings: {e}")
            self.show_error_message(f"Failed to generate UMAP embeddings: {str(e)}")
            return None

    def plot_topic_heatmap(self):
        """Plot topic similarity heatmap."""
        self.fig.clear()

        # This is a placeholder - in a real implementation, you would calculate
        # topic similarities using word distributions or embeddings
        ax = self.fig.add_subplot(111)

        # Create dummy similarity matrix for now
        n_topics = min(self.current_result.num_topics, 20)  # Limit for readability
        similarity_matrix = np.random.rand(n_topics, n_topics)
        np.fill_diagonal(similarity_matrix, 1.0)

        im = ax.imshow(similarity_matrix, cmap=self.color_scheme.get(), aspect="auto")

        ax.set_xlabel("Topic ID")
        ax.set_ylabel("Topic ID")
        ax.set_title("Topic Similarity Heatmap (Placeholder)")

        # Add colorbar
        cbar = self.fig.colorbar(im, ax=ax)
        cbar.set_label("Similarity")

        self.fig.tight_layout()
        self.canvas.draw()

    def plot_word_cloud(self):
        """Plot word cloud for selected topic or overall."""
        self.fig.clear()

        if self.current_topic_selection is not None:
            # Word cloud for specific topic
            topic = self.current_result.get_topic_by_id(self.current_topic_selection)
            if topic:
                word_freq = dict(topic.words)
                title = f"Word Cloud - Topic {topic.topic_id}"
            else:
                self.show_error_message("Selected topic not found")
                return
        else:
            # Overall word cloud from all topics
            word_freq = {}
            for topic in self.current_result.topic_info:
                if topic.topic_id != -1:  # Skip outliers
                    for word, score in topic.words:
                        word_freq[word] = word_freq.get(word, 0) + score
            title = "Word Cloud - All Topics"

        if not word_freq:
            self.show_error_message("No words available for word cloud")
            return

        # Generate word cloud
        try:
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                colormap=self.color_scheme.get(),
                max_words=100,
            ).generate_from_frequencies(word_freq)

            ax = self.fig.add_subplot(111)
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.set_title(title)
            ax.axis("off")

        except Exception as e:
            self.show_error_message(f"Failed to generate word cloud: {str(e)}")
            return

        self.fig.tight_layout()
        self.canvas.draw()

    def plot_topic_evolution(self):
        """Plot topic evolution over time (placeholder)."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Placeholder for topic evolution
        ax.text(
            0.5,
            0.5,
            "Topic evolution visualization\nrequires temporal data.\n\nThis feature will be implemented\nwhen timestamp information is available.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        self.canvas.draw()

    def show_error_message(self, message: str):
        """Show error message in plot area."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            f"Error: {message}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        self.canvas.draw()

    def on_plot_type_change(self, value):
        """Handle plot type change."""
        logger.info(f"Plot type changed to: {value}")
        self.generate_visualization()

    def on_color_change(self, value):
        """Handle color scheme change."""
        logger.info(f"Color scheme changed to: {value}")
        self.generate_visualization()

    def on_option_change(self):
        """Handle option changes."""
        self.generate_visualization()

    def on_doc_count_change(self, value):
        """Handle document count slider change."""
        doc_count = int(value)
        self.doc_count_label.configure(text=str(doc_count))

    def update_document_count(self):
        """Update the dataset with the new document count."""
        if not self.controller:
            return

        from tkinter import messagebox

        new_count = self.num_documents.get()

        # Show confirmation dialog since this will re-run topic modeling
        response = messagebox.askyesno(
            "Update Document Count",
            f"This will re-run topic modeling with {new_count} documents.\n\n"
            f"This may take several minutes. Continue?",
            icon="question",
        )

        if response:
            # Get the current data configuration and update the document limit
            # Note: This is a simplified approach - in a full implementation,
            # you'd want to coordinate with the data controller to update settings

            # For now, just show a message that this feature needs full implementation
            messagebox.showinfo(
                "Feature Coming Soon",
                f"Document count adjustment to {new_count} documents is partially implemented.\n\n"
                f"Currently showing visualizations with existing data.\n\n"
                f"Full re-processing with custom document counts will be added in the next update.",
            )

        # For now, just refresh visualizations with existing data
        self.refresh_visualizations()

    def on_plotly_plot_change(self, value):
        """Handle Plotly plot type change."""
        logger.info(f"Plotly plot type changed to: {value}")

    def _get_processed_texts(self):
        """Get the processed texts from the data controller or embedding result."""
        try:
            # Try to get from the main window's embedding result first
            if hasattr(self.parent, 'get_embedding_result'):
                embedding_result = self.parent.get_embedding_result()
                if embedding_result and embedding_result.texts:
                    return embedding_result.texts
            
            # Try to get from data controller as fallback
            if hasattr(self.parent, 'data_import_tab'):
                data_controller = self.parent.data_import_tab.get_data_controller()
                if data_controller and hasattr(data_controller, 'get_combined_texts'):
                    return data_controller.get_combined_texts()
            
            # If no texts available, return None
            return None
            
        except Exception as e:
            logger.warning(f"Could not retrieve processed texts: {e}")
            return None

    def generate_plotly_visualization(self):
        """Generate interactive Plotly visualization."""
        if not self.current_result:
            if self.has_tkinterweb:
                self.plotly_display.load_html(
                    "<html><body style='font-family: Arial; padding: 20px; text-align: center;'>"
                    "<h3>No Data Available</h3>"
                    "<p>No topic modeling results available.</p>"
                    "<p>Run topic modeling first in the Topic Modeling tab.</p>"
                    "</body></html>"
                )
            else:
                self.plotly_display.delete("1.0", "end")
                self.plotly_display.insert(
                    "1.0",
                    "No topic modeling results available.\n\nRun topic modeling first.",
                )
            return

        try:
            plot_type = self.plotly_plot_type.get()

            if plot_type == "interactive_scatter":
                html_content = self.create_plotly_scatter()
            elif plot_type == "interactive_distribution":
                html_content = self.create_plotly_distribution()
            elif plot_type == "interactive_heatmap":
                html_content = self.create_plotly_heatmap()
            else:
                html_content = "<p>Plot type not implemented yet.</p>"

            # Display HTML content
            if self.has_tkinterweb:
                self.plotly_display.load_html(html_content)
            else:
                self.plotly_display.delete("1.0", "end")
                self.plotly_display.insert("1.0", html_content)

        except Exception as e:
            logger.error(f"Error generating Plotly visualization: {e}")
            error_html = (
                "<html><body style='font-family: Arial; padding: 20px; text-align: center;'>"
                f"<h3>Error Generating Plot</h3><p>{str(e)}</p></body></html>"
            )
            if self.has_tkinterweb:
                self.plotly_display.load_html(error_html)
            else:
                self.plotly_display.delete("1.0", "end")
                self.plotly_display.insert("1.0", f"Error generating plot: {str(e)}")

    def create_plotly_scatter(self):
        """Create interactive scatter plot using Plotly."""
        try:
            import plotly.express as px

            if self.current_result.umap_embeddings is None:
                return "<p>No UMAP embeddings available for scatter plot.</p>"

            embeddings = self.current_result.umap_embeddings
            topics = self.current_result.topics

            # Get processed texts from the controller
            texts = self._get_processed_texts()
            if texts is None:
                texts = [f"Document {i}" for i in range(len(topics))]

            # Get topic keywords for hover display
            topic_keywords = {}
            for topic_info in self.current_result.topic_info:
                if len(topic_info.words) > 0:
                    keywords = [word for word, _ in topic_info.words[:5]]
                    topic_keywords[topic_info.topic_id] = ", ".join(keywords)
                else:
                    topic_keywords[topic_info.topic_id] = "No keywords"

            # Create DataFrame for Plotly
            import pandas as pd

            # Truncate text snippets for hover display (max 150 chars)
            text_snippets = [
                text[:150] + "..." if len(text) > 150 else text 
                for text in texts
            ]

            # Get topic keywords for each document
            doc_keywords = [
                topic_keywords.get(topic, "No keywords") if topic != -1 else "Outlier"
                for topic in topics
            ]

            df = pd.DataFrame(
                {
                    "UMAP_1": embeddings[:, 0],
                    "UMAP_2": embeddings[:, 1],
                    "Topic": topics,
                    "Topic_Label": [
                        f"Topic {t}" if t != -1 else "Outlier" for t in topics
                    ],
                    "Document_ID": [f"Doc {i}" for i in range(len(topics))],
                    "Keywords": doc_keywords,
                    "Text_Snippet": text_snippets,
                }
            )

            # Create scatter plot with enhanced hover information
            fig = px.scatter(
                df,
                x="UMAP_1",
                y="UMAP_2",
                color="Topic_Label",
                title="Interactive Document Embeddings",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )

            # Customize hover template for better display
            fig.update_traces(
                hovertemplate="<br>".join([
                    "<b>%{customdata[0]}</b>",  # Document ID
                    "<b>Topic %{customdata[1]}</b>",  # Topic number
                    "<i>Keywords:</i> %{customdata[2]}",  # Topic keywords
                    "<i>Text:</i> %{customdata[3]}",  # Text snippet
                    "<extra></extra>"  # Remove trace box
                ]),
                customdata=list(zip(
                    df["Document_ID"],
                    df["Topic"],
                    df["Keywords"],
                    df["Text_Snippet"]
                ))
            )

            fig.update_layout(width=800, height=600, showlegend=True)

            return fig.to_html(include_plotlyjs="cdn")

        except ImportError:
            return "<p>Plotly not installed. Install with: pip install plotly</p>"
        except Exception as e:
            return f"<p>Error creating scatter plot: {str(e)}</p>"

    def create_plotly_distribution(self):
        """Create interactive topic distribution plot."""
        try:
            import plotly.graph_objects as go

            # Get topic data with keywords and details
            topic_names = []
            topic_sizes = []
            topic_keywords = []
            topic_percentages = []
            
            total_docs = len(self.current_result.topics)
            
            for topic in self.current_result.topic_info:
                if topic.topic_id != -1:
                    topic_names.append(f"Topic {topic.topic_id}")
                    topic_sizes.append(topic.size)
                    
                    # Get top 5 keywords for hover
                    if len(topic.words) > 0:
                        keywords = [word for word, _ in topic.words[:5]]
                        topic_keywords.append(", ".join(keywords))
                    else:
                        topic_keywords.append("No keywords")
                    
                    # Calculate percentage
                    percentage = (topic.size / total_docs) * 100 if total_docs > 0 else 0
                    topic_percentages.append(f"{percentage:.1f}%")

            if not topic_names:
                return "<p>No topics available for distribution plot.</p>"

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=topic_names,
                        y=topic_sizes,
                        marker_color="lightblue",
                        text=topic_sizes,
                        textposition="auto",
                        hovertemplate="<br>".join([
                            "<b>%{x}</b>",
                            "Documents: %{y}",
                            "Percentage: %{customdata[1]}",
                            "<i>Keywords:</i> %{customdata[0]}",
                            "<extra></extra>"
                        ]),
                        customdata=list(zip(topic_keywords, topic_percentages))
                    )
                ]
            )

            fig.update_layout(
                title="Interactive Topic Distribution",
                xaxis_title="Topics",
                yaxis_title="Number of Documents",
                width=800,
                height=500,
            )

            return fig.to_html(include_plotlyjs="cdn")

        except ImportError:
            return "<p>Plotly not installed. Install with: pip install plotly</p>"
        except Exception as e:
            return f"<p>Error creating distribution plot: {str(e)}</p>"

    def create_plotly_heatmap(self):
        """Create interactive topic similarity heatmap."""
        try:
            import plotly.graph_objects as go
            import numpy as np

            # Get topics (excluding outliers)
            valid_topics = [t for t in self.current_result.topic_info if t.topic_id != -1]
            num_topics = len(valid_topics)

            if num_topics < 2:
                return "<p>Need at least 2 topics for heatmap.</p>"

            # Create sample correlation matrix (in real implementation, this would be topic similarities)
            np.random.seed(42)
            correlation_matrix = np.random.rand(num_topics, num_topics)
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
            np.fill_diagonal(correlation_matrix, 1.0)

            # Get topic labels and keywords
            topic_labels = []
            topic_keywords_map = {}
            
            for i, topic in enumerate(valid_topics):
                label = f"Topic {topic.topic_id}"
                topic_labels.append(label)
                
                # Get top 3 keywords for hover
                if len(topic.words) > 0:
                    keywords = [word for word, _ in topic.words[:3]]
                    topic_keywords_map[i] = ", ".join(keywords)
                else:
                    topic_keywords_map[i] = "No keywords"

            # Create hover text matrix with topic keywords
            hover_text = []
            for i in range(num_topics):
                row = []
                for j in range(num_topics):
                    if i == j:
                        text = f"<b>{topic_labels[i]}</b><br>Similarity: {correlation_matrix[i,j]:.2f}<br>Keywords: {topic_keywords_map[i]}"
                    else:
                        text = f"<b>{topic_labels[i]} vs {topic_labels[j]}</b><br>Similarity: {correlation_matrix[i,j]:.2f}<br>Keywords:<br>{topic_labels[i]}: {topic_keywords_map[i]}<br>{topic_labels[j]}: {topic_keywords_map[j]}"
                    row.append(text)
                hover_text.append(row)

            fig = go.Figure(
                data=go.Heatmap(
                    z=correlation_matrix,
                    x=topic_labels,
                    y=topic_labels,
                    colorscale="Viridis",
                    text=np.round(correlation_matrix, 2),
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    hoverongaps=False,
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=hover_text
                )
            )

            fig.update_layout(
                title="Interactive Topic Similarity Heatmap",
                width=700,
                height=600,
                xaxis_title="Topics",
                yaxis_title="Topics"
            )

            return fig.to_html(include_plotlyjs="cdn")

        except ImportError:
            return "<p>Plotly not installed. Install with: pip install plotly</p>"
        except Exception as e:
            return f"<p>Error creating heatmap: {str(e)}</p>"

    def export_plotly_html(self):
        """Export current Plotly visualization as HTML file."""
        if self.has_tkinterweb:
            # For tkinterweb, we need to regenerate the HTML content
            try:
                plot_type = self.plotly_plot_type.get()
                if plot_type == "interactive_scatter":
                    html_content = self.create_plotly_scatter()
                elif plot_type == "interactive_distribution":
                    html_content = self.create_plotly_distribution()
                elif plot_type == "interactive_heatmap":
                    html_content = self.create_plotly_heatmap()
                else:
                    html_content = "<p>No plot available for export.</p>"
            except Exception as e:
                from tkinter import messagebox

                messagebox.showerror(
                    "Export Error", f"Failed to generate plot for export:\n{str(e)}"
                )
                return
        else:
            # For text widget
            html_content = self.plotly_display.get("1.0", "end-1c")

        if (
            not html_content
            or "Error" in html_content
            or "No topic" in html_content
            or "No plot" in html_content
        ):
            from tkinter import messagebox

            messagebox.showwarning(
                "Export Warning", "No valid plot to export. Generate a plot first."
            )
            return

        from tkinter import filedialog

        filename = filedialog.asksaveasfilename(
            title="Export Interactive Plot",
            defaultextension=".html",
            filetypes=[("HTML files", "*.html")],
        )

        if filename:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(html_content)

                from tkinter import messagebox

                messagebox.showinfo(
                    "Export Success", f"Interactive plot exported to:\n{filename}"
                )

            except Exception as e:
                from tkinter import messagebox

                messagebox.showerror(
                    "Export Error", f"Failed to export plot:\n{str(e)}"
                )

    def on_topic_selection_change(self, value):
        """Handle topic selection change."""
        if value == "None" or not self.current_result:
            self.current_topic_selection = None
            self.topic_info_text.delete("1.0", "end")
            self.topic_info_text.insert(
                "1.0", "Select a topic to view detailed information..."
            )
            return

        # Extract topic ID from selection
        try:
            topic_id = int(value.split(":")[0].replace("Topic ", ""))
            self.current_topic_selection = topic_id

            # Update topic info display
            topic = self.current_result.get_topic_by_id(topic_id)
            if topic:
                info_text = f"""Topic {topic.topic_id} Information
================================

📊 Size: {topic.size} documents ({topic.percentage:.1f}%)
🏷️ Name: {topic.name or f'Topic {topic.topic_id}'}

🔤 Top Words:
"""
                for i, (word, score) in enumerate(topic.words[:10], 1):
                    info_text += f"  {i:2d}. {word} ({score:.3f})\n"

                if topic.representative_docs:
                    info_text += "\n📄 Representative Documents:\n"
                    for i, doc in enumerate(topic.representative_docs[:3], 1):
                        preview = doc[:100] + "..." if len(doc) > 100 else doc
                        info_text += f"  {i}. {preview}\n\n"

                self.topic_info_text.delete("1.0", "end")
                self.topic_info_text.insert("1.0", info_text)

            # Regenerate visualization if it's word cloud (topic-specific)
            if self.plot_type.get() == "word_cloud":
                self.generate_visualization()

        except (ValueError, IndexError) as e:
            logger.error(f"Failed to parse topic selection: {e}")

    def export_current_plot(self):
        """Export current plot to file."""
        if not self.current_result:
            logger.warning("No results available for export")
            return

        from tkinter import filedialog

        filename = filedialog.asksaveasfilename(
            title="Export Plot",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("JPG files", "*.jpg"),
            ],
        )

        if filename:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches="tight")
                logger.info(f"Plot exported to: {filename}")

                # Show success message
                from tkinter import messagebox

                messagebox.showinfo("Export Success", f"Plot saved to:\n{filename}")

            except Exception as e:
                logger.error(f"Failed to export plot: {e}")
                from tkinter import messagebox

                messagebox.showerror(
                    "Export Failed", f"Failed to export plot:\n{str(e)}"
                )
