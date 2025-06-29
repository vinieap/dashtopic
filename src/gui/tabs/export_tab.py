"""
Export Tab - Comprehensive data and visualization export functionality.
"""

import logging
import os
from pathlib import Path
import customtkinter as ctk
from typing import Optional
import pandas as pd
import json
from datetime import datetime
import threading

from ...models import TopicResult
from ...controllers import TopicModelingController, DataController, EmbeddingController

logger = logging.getLogger(__name__)


class ExportTab:
    """Tab for exporting results, visualizations, and creating reports."""

    def __init__(self, parent):
        self.parent = parent
        self.topic_controller: Optional[TopicModelingController] = None
        self.data_controller: Optional[DataController] = None
        self.embedding_controller: Optional[EmbeddingController] = None

        # Export options
        self.export_data = ctk.BooleanVar(value=True)
        self.export_topics = ctk.BooleanVar(value=True)
        self.export_embeddings = ctk.BooleanVar(value=False)
        self.export_visualizations = ctk.BooleanVar(value=True)
        self.export_model = ctk.BooleanVar(value=False)
        self.export_config = ctk.BooleanVar(value=True)

        # Export format
        self.data_format = ctk.StringVar(value="excel")
        self.viz_format = ctk.StringVar(value="png")

        # State
        self.current_export_path = ctk.StringVar(value="")
        self.is_exporting = False

        self.setup_ui()
        logger.info("Export tab initialized")

    def set_controllers(
        self,
        topic_controller: TopicModelingController,
        data_controller: DataController,
        embedding_controller: EmbeddingController,
    ):
        """Set the required controllers."""
        self.topic_controller = topic_controller
        self.data_controller = data_controller
        self.embedding_controller = embedding_controller
        logger.info("Export tab controllers configured")

    def setup_ui(self):
        """Set up the tab user interface."""
        # Main content frame
        self.main_frame = ctk.CTkFrame(self.parent)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Export Results & Reports",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.title_label.pack(pady=(10, 20))

        # Export options section
        self.setup_export_options()

        # Format selection section
        self.setup_format_options()

        # Output location section
        self.setup_output_location()

        # Export controls section
        self.setup_export_controls()

        # Progress and status section
        self.setup_progress_section()

        # Preview section
        self.setup_preview_section()

    def setup_export_options(self):
        """Setup export options selection."""
        options_frame = ctk.CTkFrame(self.main_frame)
        options_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            options_frame, text="Export Options", font=ctk.CTkFont(weight="bold")
        ).pack(pady=5)

        # Options grid
        options_grid = ctk.CTkFrame(options_frame)
        options_grid.pack(fill="x", padx=10, pady=5)

        # Row 1: Data and Topics
        row1_frame = ctk.CTkFrame(options_grid)
        row1_frame.pack(fill="x", pady=2)

        self.data_check = ctk.CTkCheckBox(
            row1_frame,
            text="Original Data with Topic Assignments",
            variable=self.export_data,
        )
        self.data_check.pack(side="left", padx=10, pady=5)

        self.topics_check = ctk.CTkCheckBox(
            row1_frame,
            text="Topic Information & Statistics",
            variable=self.export_topics,
        )
        self.topics_check.pack(side="left", padx=10, pady=5)

        # Row 2: Embeddings and Visualizations
        row2_frame = ctk.CTkFrame(options_grid)
        row2_frame.pack(fill="x", pady=2)

        self.embeddings_check = ctk.CTkCheckBox(
            row2_frame,
            text="Document Embeddings (UMAP)",
            variable=self.export_embeddings,
        )
        self.embeddings_check.pack(side="left", padx=10, pady=5)

        self.viz_check = ctk.CTkCheckBox(
            row2_frame,
            text="Visualizations & Plots",
            variable=self.export_visualizations,
        )
        self.viz_check.pack(side="left", padx=10, pady=5)

        # Row 3: Model and Config
        row3_frame = ctk.CTkFrame(options_grid)
        row3_frame.pack(fill="x", pady=2)

        self.model_check = ctk.CTkCheckBox(
            row3_frame,
            text="Trained Model (for future use)",
            variable=self.export_model,
        )
        self.model_check.pack(side="left", padx=10, pady=5)

        self.config_check = ctk.CTkCheckBox(
            row3_frame, text="Configuration Settings", variable=self.export_config
        )
        self.config_check.pack(side="left", padx=10, pady=5)

    def setup_format_options(self):
        """Setup format selection options."""
        format_frame = ctk.CTkFrame(self.main_frame)
        format_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            format_frame, text="Export Formats", font=ctk.CTkFont(weight="bold")
        ).pack(pady=5)

        # Format controls
        format_controls = ctk.CTkFrame(format_frame)
        format_controls.pack(fill="x", padx=10, pady=5)

        # Data format
        ctk.CTkLabel(format_controls, text="Data Format:").pack(
            side="left", padx=(10, 5)
        )
        self.data_format_menu = ctk.CTkOptionMenu(
            format_controls,
            variable=self.data_format,
            values=["excel", "csv", "parquet", "json"],
            width=100,
        )
        self.data_format_menu.pack(side="left", padx=5)

        # Visualization format
        ctk.CTkLabel(format_controls, text="Visualization Format:").pack(
            side="left", padx=(20, 5)
        )
        self.viz_format_menu = ctk.CTkOptionMenu(
            format_controls,
            variable=self.viz_format,
            values=["png", "pdf", "svg", "jpg", "html"],
            width=100,
        )
        self.viz_format_menu.pack(side="left", padx=5)

    def setup_output_location(self):
        """Setup output location selection."""
        location_frame = ctk.CTkFrame(self.main_frame)
        location_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            location_frame, text="Output Location", font=ctk.CTkFont(weight="bold")
        ).pack(pady=5)

        # Location controls
        location_controls = ctk.CTkFrame(location_frame)
        location_controls.pack(fill="x", padx=10, pady=5)

        self.location_entry = ctk.CTkEntry(
            location_controls,
            textvariable=self.current_export_path,
            placeholder_text="Select export directory...",
        )
        self.location_entry.pack(side="left", fill="x", expand=True, padx=(10, 5))

        self.browse_btn = ctk.CTkButton(
            location_controls,
            text="Browse",
            command=self.browse_export_location,
            width=80,
        )
        self.browse_btn.pack(side="right", padx=(5, 10))

    def setup_export_controls(self):
        """Setup export control buttons."""
        controls_frame = ctk.CTkFrame(self.main_frame)
        controls_frame.pack(fill="x", pady=(0, 10))

        # Control buttons
        buttons_frame = ctk.CTkFrame(controls_frame)
        buttons_frame.pack(pady=10)

        # Quick export buttons
        self.quick_excel_btn = ctk.CTkButton(
            buttons_frame,
            text="Quick Export to Excel",
            command=self.quick_export_excel,
            width=150,
        )
        self.quick_excel_btn.pack(side="left", padx=5)

        self.quick_report_btn = ctk.CTkButton(
            buttons_frame,
            text="Generate HTML Report",
            command=self.generate_html_report,
            width=150,
        )
        self.quick_report_btn.pack(side="left", padx=5)

        # Main export button
        self.export_btn = ctk.CTkButton(
            buttons_frame,
            text="Export All Selected",
            command=self.start_full_export,
            width=150,
            fg_color="green",
            hover_color="darkgreen",
        )
        self.export_btn.pack(side="left", padx=5)

        # Clear button
        self.clear_btn = ctk.CTkButton(
            buttons_frame,
            text="Clear Settings",
            command=self.clear_settings,
            width=100,
            fg_color="gray",
            hover_color="darkgray",
        )
        self.clear_btn.pack(side="left", padx=5)

    def setup_progress_section(self):
        """Setup progress tracking section."""
        progress_frame = ctk.CTkFrame(self.main_frame)
        progress_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            progress_frame, text="Export Progress", font=ctk.CTkFont(weight="bold")
        ).pack(pady=5)

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(fill="x", padx=10, pady=5)
        self.progress_bar.set(0)

        # Status label
        self.status_label = ctk.CTkLabel(progress_frame, text="Ready to export")
        self.status_label.pack(pady=5)

    def setup_preview_section(self):
        """Setup export preview section."""
        preview_frame = ctk.CTkFrame(self.main_frame)
        preview_frame.pack(fill="both", expand=True, pady=(0, 10))

        ctk.CTkLabel(
            preview_frame, text="Export Preview", font=ctk.CTkFont(weight="bold")
        ).pack(pady=5)

        # Preview text area
        self.preview_text = ctk.CTkTextbox(preview_frame, height=150)
        self.preview_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Update preview button
        self.update_preview_btn = ctk.CTkButton(
            preview_frame, text="Update Preview", command=self.update_preview, width=120
        )
        self.update_preview_btn.pack(pady=(0, 10))

        # Initialize preview
        self.update_preview()

    def browse_export_location(self):
        """Browse for export directory."""
        from tkinter import filedialog

        directory = filedialog.askdirectory(
            title="Select Export Directory", initialdir=os.getcwd()
        )

        if directory:
            self.current_export_path.set(directory)
            logger.info(f"Export directory set to: {directory}")

    def update_preview(self):
        """Update export preview."""
        preview_text = "Export Preview\n" + "=" * 50 + "\n\n"

        # Check data availability
        has_topic_results = (
            self.topic_controller and self.topic_controller.get_current_result()
        )
        has_data = self.data_controller and self.data_controller.get_current_config()
        has_embeddings = (
            self.embedding_controller and self.embedding_controller.get_current_result()
        )

        if not any([has_topic_results, has_data, has_embeddings]):
            preview_text += "⚠️ No data available for export.\n"
            preview_text += "Please run the topic modeling workflow first.\n"
        else:
            export_path = self.current_export_path.get() or os.getcwd()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_folder = f"bertopic_export_{timestamp}"

            preview_text += f"📁 Export Location: {export_path}\n"
            preview_text += f"📁 Export Folder: {export_folder}\n\n"

            preview_text += "📄 Files to be created:\n" + "-" * 30 + "\n"

            if self.export_data.get() and has_data:
                ext = self.data_format.get()
                preview_text += (
                    f"• data_with_topics.{ext} - Original data with topic assignments\n"
                )

            if self.export_topics.get() and has_topic_results:
                ext = self.data_format.get()
                preview_text += (
                    f"• topic_info.{ext} - Topic information and statistics\n"
                )
                preview_text += f"• topic_words.{ext} - Top words for each topic\n"

            if self.export_embeddings.get() and has_embeddings:
                preview_text += (
                    "• embeddings.npz - Document embeddings (numpy format)\n"
                )
                preview_text += "• umap_embeddings.npz - UMAP reduced embeddings\n"

            if self.export_visualizations.get() and has_topic_results:
                ext = self.viz_format.get()
                preview_text += (
                    f"• topic_distribution.{ext} - Topic size distribution\n"
                )
                preview_text += (
                    f"• document_scatter.{ext} - Document embedding scatter plot\n"
                )
                preview_text += (
                    "• word_clouds/ - Folder with word clouds for each topic\n"
                )

            if self.export_model.get() and has_topic_results:
                preview_text += "• bertopic_model/ - Trained BERTopic model folder\n"

            if self.export_config.get():
                preview_text += (
                    "• export_config.json - Export configuration and metadata\n"
                )

            preview_text += "\n📊 Summary:\n"
            if has_topic_results:
                result = self.topic_controller.get_current_result()
                preview_text += f"• Topics: {result.num_topics}\n"
                preview_text += f"• Documents: {result.num_documents}\n"
                preview_text += f"• Outliers: {result.outlier_percentage:.1f}%\n"

        self.preview_text.delete("1.0", "end")
        self.preview_text.insert("1.0", preview_text)

    def quick_export_excel(self):
        """Quick export to Excel format."""
        if not self.topic_controller or not self.topic_controller.get_current_result():
            self.show_error("No topic modeling results available")
            return

        from tkinter import filedialog

        filename = filedialog.asksaveasfilename(
            title="Export to Excel",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
        )

        if filename:
            self.perform_quick_excel_export(filename)

    def perform_quick_excel_export(self, filename: str):
        """Perform quick Excel export in background thread."""

        def export_worker():
            try:
                self.update_progress(0.1, "Starting Excel export...")

                result = self.topic_controller.get_current_result()

                with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                    # Topic information
                    self.update_progress(0.3, "Exporting topic information...")
                    topic_data = []
                    for topic in result.topic_info:
                        topic_data.append(
                            {
                                "Topic_ID": topic.topic_id,
                                "Size": topic.size,
                                "Percentage": topic.percentage,
                                "Top_Words": topic.top_words_string,
                                "Name": topic.name or f"Topic {topic.topic_id}",
                            }
                        )

                    df_topics = pd.DataFrame(topic_data)
                    df_topics.to_excel(writer, sheet_name="Topic_Summary", index=False)

                    # Document assignments
                    self.update_progress(0.6, "Exporting document assignments...")
                    doc_data = {
                        "Document_Index": range(len(result.topics)),
                        "Topic_ID": result.topics,
                    }

                    if result.probabilities:
                        for i, topic in enumerate(result.topic_info):
                            if topic.topic_id != -1:
                                doc_data[f"Topic_{topic.topic_id}_Probability"] = [
                                    prob[i] if i < len(prob) else 0.0
                                    for prob in result.probabilities
                                ]

                    df_docs = pd.DataFrame(doc_data)
                    df_docs.to_excel(writer, sheet_name="Document_Topics", index=False)

                    # Quality metrics
                    self.update_progress(0.8, "Adding quality metrics...")
                    metrics_data = {
                        "Metric": [
                            "Number of Topics",
                            "Total Documents",
                            "Outlier Percentage",
                            "Training Time (seconds)",
                            "Silhouette Score",
                            "Calinski-Harabasz Score",
                            "Davies-Bouldin Score",
                        ],
                        "Value": [
                            result.num_topics,
                            result.num_documents,
                            result.outlier_percentage,
                            result.training_time_seconds,
                            result.silhouette_score or "N/A",
                            result.calinski_harabasz_score or "N/A",
                            result.davies_bouldin_score or "N/A",
                        ],
                    }
                    df_metrics = pd.DataFrame(metrics_data)
                    df_metrics.to_excel(
                        writer, sheet_name="Quality_Metrics", index=False
                    )

                self.update_progress(1.0, f"Excel export completed: {filename}")

                # Show success message
                from tkinter import messagebox

                self.parent.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Export Success", f"Data exported successfully to:\n{filename}"
                    ),
                )

            except Exception as e:
                logger.error(f"Excel export failed: {e}")
                error_msg = f"Excel export failed: {str(e)}"
                self.parent.after(0, lambda: self.show_error(error_msg))
            finally:
                self.parent.after(0, lambda: self.set_export_state(False))

        self.set_export_state(True)
        thread = threading.Thread(target=export_worker, daemon=True)
        thread.start()

    def generate_html_report(self):
        """Generate comprehensive HTML report."""
        if not self.topic_controller or not self.topic_controller.get_current_result():
            self.show_error("No topic modeling results available")
            return

        from tkinter import filedialog

        filename = filedialog.asksaveasfilename(
            title="Generate HTML Report",
            defaultextension=".html",
            filetypes=[("HTML files", "*.html")],
        )

        if filename:
            self.perform_html_report_generation(filename)

    def perform_html_report_generation(self, filename: str):
        """Generate HTML report in background thread."""

        def report_worker():
            try:
                self.update_progress(0.1, "Starting HTML report generation...")

                result = self.topic_controller.get_current_result()

                # Generate HTML content
                html_content = self.generate_html_content(result)

                self.update_progress(0.8, "Writing HTML file...")

                with open(filename, "w", encoding="utf-8") as f:
                    f.write(html_content)

                self.update_progress(1.0, f"HTML report completed: {filename}")

                # Show success message
                from tkinter import messagebox

                self.parent.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Report Generated",
                        f"HTML report generated successfully:\n{filename}\n\nOpen in browser to view.",
                    ),
                )

            except Exception as e:
                logger.error(f"HTML report generation failed: {e}")
                error_msg = f"HTML report generation failed: {str(e)}"
                self.parent.after(0, lambda: self.show_error(error_msg))
            finally:
                self.parent.after(0, lambda: self.set_export_state(False))

        self.set_export_state(True)
        thread = threading.Thread(target=report_worker, daemon=True)
        thread.start()

    def generate_html_content(self, result: TopicResult) -> str:
        """Generate HTML content for the report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BERTopic Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metric-box {{ display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        .metric-label {{ font-size: 12px; color: #7f8c8d; }}
        .topic-item {{ margin: 15px 0; padding: 15px; background: #f8f9fa; border-left: 4px solid #3498db; border-radius: 5px; }}
        .topic-words {{ font-style: italic; color: #555; }}
        .quality-metrics {{ display: flex; flex-wrap: wrap; gap: 20px; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 BERTopic Analysis Report</h1>
        <p><strong>Generated:</strong> {timestamp}</p>
        
        <h2>📊 Overview</h2>
        <div class="quality-metrics">
            <div class="metric-box">
                <div class="metric-value">{result.num_topics}</div>
                <div class="metric-label">Topics Found</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{result.num_documents}</div>
                <div class="metric-label">Documents</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{result.outlier_percentage:.1f}%</div>
                <div class="metric-label">Outliers</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{result.training_time_seconds:.1f}s</div>
                <div class="metric-label">Training Time</div>
            </div>
        </div>
        
        <h2>📈 Quality Metrics</h2>
        <ul>
"""

        if result.silhouette_score:
            html += f"<li><strong>Silhouette Score:</strong> {result.silhouette_score:.3f}</li>"
        if result.calinski_harabasz_score:
            html += f"<li><strong>Calinski-Harabasz Score:</strong> {result.calinski_harabasz_score:.2f}</li>"
        if result.davies_bouldin_score:
            html += f"<li><strong>Davies-Bouldin Score:</strong> {result.davies_bouldin_score:.3f}</li>"

        html += """
        </ul>
        
        <h2>🏷️ Topics</h2>
"""

        # Add topics
        for topic in sorted(result.topic_info, key=lambda x: x.size, reverse=True):
            if topic.topic_id == -1:
                continue

            html += f"""
        <div class="topic-item">
            <h3>Topic {topic.topic_id}</h3>
            <p><strong>Size:</strong> {topic.size} documents ({topic.percentage:.1f}%)</p>
            <p class="topic-words"><strong>Key Words:</strong> {topic.top_words_string}</p>
        </div>
"""

        html += """
        <div class="footer">
            <p>Report generated by BERTopic Desktop Application</p>
            <p>For more detailed analysis, export the full dataset or use the interactive visualizations.</p>
        </div>
    </div>
</body>
</html>
"""

        return html

    def start_full_export(self):
        """Start full export with all selected options."""
        if not self.current_export_path.get():
            self.show_error("Please select an export directory first")
            return

        # Validate that we have data to export
        has_data = False
        if self.export_data.get() or self.export_topics.get():
            has_data = (
                self.topic_controller and self.topic_controller.get_current_result()
            )

        if not has_data:
            self.show_error("No topic modeling results available for export")
            return

        self.perform_full_export()

    def perform_full_export(self):
        """Perform full export in background thread."""

        def export_worker():
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_folder = (
                    Path(self.current_export_path.get())
                    / f"bertopic_export_{timestamp}"
                )
                export_folder.mkdir(parents=True, exist_ok=True)

                self.update_progress(
                    0.1, f"Created export folder: {export_folder.name}"
                )

                result = self.topic_controller.get_current_result()

                # Export data with topics
                if self.export_data.get():
                    self.update_progress(
                        0.2, "Exporting data with topic assignments..."
                    )
                    self.export_data_with_topics(export_folder, result)

                # Export topic information
                if self.export_topics.get():
                    self.update_progress(0.4, "Exporting topic information...")
                    self.export_topic_information(export_folder, result)

                # Export embeddings
                if self.export_embeddings.get():
                    self.update_progress(0.6, "Exporting embeddings...")
                    self.export_embeddings_data(export_folder, result)

                # Export visualizations
                if self.export_visualizations.get():
                    self.update_progress(0.8, "Exporting visualizations...")
                    self.export_visualizations_data(export_folder, result)

                # Export configuration
                if self.export_config.get():
                    self.update_progress(0.9, "Exporting configuration...")
                    self.export_configuration(export_folder, result)

                self.update_progress(1.0, "Export completed successfully!")

                # Show success message
                from tkinter import messagebox

                self.parent.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Export Complete",
                        f"All selected data exported successfully to:\n{export_folder}",
                    ),
                )

            except Exception as e:
                logger.error(f"Full export failed: {e}")
                error_msg = f"Export failed: {str(e)}"
                self.parent.after(0, lambda: self.show_error(error_msg))
            finally:
                self.parent.after(0, lambda: self.set_export_state(False))

        self.set_export_state(True)
        thread = threading.Thread(target=export_worker, daemon=True)
        thread.start()

    def export_data_with_topics(self, export_folder: Path, result: TopicResult):
        """Export original data with topic assignments."""
        # This is a simplified version - in a real implementation,
        # you would get the original data from the data controller
        doc_data = {
            "Document_Index": range(len(result.topics)),
            "Topic_ID": result.topics,
        }

        df = pd.DataFrame(doc_data)

        format_ext = self.data_format.get()
        filename = export_folder / f"data_with_topics.{format_ext}"

        if format_ext == "excel":
            df.to_excel(filename, index=False)
        elif format_ext == "csv":
            df.to_csv(filename, index=False)
        elif format_ext == "parquet":
            df.to_parquet(filename)
        elif format_ext == "json":
            df.to_json(filename, orient="records", indent=2)

    def export_topic_information(self, export_folder: Path, result: TopicResult):
        """Export topic information and statistics."""
        topic_data = []
        for topic in result.topic_info:
            topic_data.append(
                {
                    "Topic_ID": topic.topic_id,
                    "Size": topic.size,
                    "Percentage": topic.percentage,
                    "Top_Words": topic.top_words_string,
                    "Name": topic.name or f"Topic {topic.topic_id}",
                }
            )

        df = pd.DataFrame(topic_data)

        format_ext = self.data_format.get()
        filename = export_folder / f"topic_information.{format_ext}"

        if format_ext == "excel":
            df.to_excel(filename, index=False)
        elif format_ext == "csv":
            df.to_csv(filename, index=False)
        elif format_ext == "parquet":
            df.to_parquet(filename)
        elif format_ext == "json":
            df.to_json(filename, orient="records", indent=2)

    def export_embeddings_data(self, export_folder: Path, result: TopicResult):
        """Export embedding data."""
        import numpy as np

        if result.embeddings is not None:
            np.savez_compressed(
                export_folder / "embeddings.npz", embeddings=result.embeddings
            )

        if result.umap_embeddings is not None:
            np.savez_compressed(
                export_folder / "umap_embeddings.npz",
                umap_embeddings=result.umap_embeddings,
            )

    def export_visualizations_data(self, export_folder: Path, result: TopicResult):
        """Export visualization data (placeholder)."""
        # This would integrate with the visualization tab to export plots
        # For now, create a placeholder file
        viz_folder = export_folder / "visualizations"
        viz_folder.mkdir(exist_ok=True)

        with open(viz_folder / "README.txt", "w") as f:
            f.write("Visualization exports would be saved here.\n")
            f.write("This feature integrates with the Visualization tab.\n")

    def export_configuration(self, export_folder: Path, result: TopicResult):
        """Export configuration and metadata."""
        config_data = {
            "export_timestamp": datetime.now().isoformat(),
            "export_options": {
                "data": self.export_data.get(),
                "topics": self.export_topics.get(),
                "embeddings": self.export_embeddings.get(),
                "visualizations": self.export_visualizations.get(),
                "model": self.export_model.get(),
                "config": self.export_config.get(),
            },
            "data_format": self.data_format.get(),
            "visualization_format": self.viz_format.get(),
            "results_summary": {
                "num_topics": result.num_topics,
                "num_documents": result.num_documents,
                "outlier_percentage": result.outlier_percentage,
                "training_time_seconds": result.training_time_seconds,
                "silhouette_score": result.silhouette_score,
                "calinski_harabasz_score": result.calinski_harabasz_score,
                "davies_bouldin_score": result.davies_bouldin_score,
            },
        }

        if result.config:
            config_data["topic_model_config"] = {
                "top_k_words": result.config.top_k_words,
                "calculate_probabilities": result.config.calculate_probabilities,
                "save_model": result.config.save_model,
                # Add more config details as needed
            }

        with open(export_folder / "export_metadata.json", "w") as f:
            json.dump(config_data, f, indent=2, default=str)

    def clear_settings(self):
        """Clear all export settings."""
        self.export_data.set(True)
        self.export_topics.set(True)
        self.export_embeddings.set(False)
        self.export_visualizations.set(True)
        self.export_model.set(False)
        self.export_config.set(True)
        self.data_format.set("excel")
        self.viz_format.set("png")
        self.current_export_path.set("")
        self.update_preview()
        logger.info("Export settings cleared")

    def update_progress(self, progress: float, message: str):
        """Update progress bar and status."""

        def update_ui():
            self.progress_bar.set(progress)
            self.status_label.configure(text=message)

        if self.parent:
            self.parent.after(0, update_ui)

    def set_export_state(self, is_exporting: bool):
        """Set export state and update UI."""
        self.is_exporting = is_exporting

        # Enable/disable buttons
        state = "disabled" if is_exporting else "normal"
        self.export_btn.configure(state=state)
        self.quick_excel_btn.configure(state=state)
        self.quick_report_btn.configure(state=state)

        if not is_exporting:
            self.progress_bar.set(0)
            self.status_label.configure(text="Ready to export")

    def show_error(self, message: str):
        """Show error message."""
        from tkinter import messagebox

        messagebox.showerror("Export Error", message)
        logger.error(f"Export error: {message}")
        self.status_label.configure(text=f"Error: {message}")
