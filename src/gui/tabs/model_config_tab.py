"""
Model Configuration Tab - Embedding model and clustering configuration.
"""

import logging
from typing import Optional, List
import customtkinter as ctk
from tkinter import messagebox
import threading

from ...models.data_models import ModelInfo, EmbeddingConfig, EmbeddingResult
from ...controllers.embedding_controller import EmbeddingController

logger = logging.getLogger(__name__)


class ModelConfigTab:
    """Tab for configuring embedding models and clustering parameters."""

    def __init__(self, parent, embedding_controller=None):
        self.parent = parent
        self.embedding_controller = embedding_controller or EmbeddingController()

        # State variables
        self.available_models: List[ModelInfo] = []
        self.selected_model: Optional[ModelInfo] = None
        self.embedding_config = EmbeddingConfig()
        self.current_embedding_result: Optional[EmbeddingResult] = None

        self.setup_ui()
        self.load_models()
        logger.info("Model Configuration tab initialized")

    def setup_ui(self):
        """Set up the tab user interface."""
        # Main scrollable frame
        self.main_frame = ctk.CTkScrollableFrame(self.parent)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Model Configuration",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.title_label.pack(pady=(10, 20))

        # Create sections
        self.setup_model_selection_section()
        self.setup_model_info_section()
        self.setup_embedding_config_section()
        self.setup_cache_management_section()
        self.setup_actions_section()

    def setup_model_selection_section(self):
        """Setup model selection section."""
        # Model Selection Frame
        self.model_selection_frame = ctk.CTkFrame(self.main_frame)
        self.model_selection_frame.pack(fill="x", padx=10, pady=(0, 15))

        # Section title
        section_title = ctk.CTkLabel(
            self.model_selection_frame,
            text="Model Selection",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        section_title.pack(pady=(15, 10))

        # Model selection controls frame
        selection_controls_frame = ctk.CTkFrame(self.model_selection_frame)
        selection_controls_frame.pack(fill="x", padx=15, pady=(0, 15))

        # Refresh models button
        self.refresh_button = ctk.CTkButton(
            selection_controls_frame,
            text="🔄 Refresh Models",
            command=self.refresh_models,
            width=150,
        )
        self.refresh_button.pack(side="left", padx=5, pady=10)

        # Loading indicator
        self.loading_label = ctk.CTkLabel(
            selection_controls_frame, text="", font=ctk.CTkFont(size=12)
        )
        self.loading_label.pack(side="left", padx=(20, 5), pady=10)

        # Model list frame
        self.model_list_frame = ctk.CTkFrame(self.model_selection_frame)
        self.model_list_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        # Model list header
        header_frame = ctk.CTkFrame(self.model_list_frame)
        header_frame.pack(fill="x", padx=5, pady=(10, 5))

        ctk.CTkLabel(
            header_frame, text="Model Name", font=ctk.CTkFont(weight="bold")
        ).pack(side="left", padx=10)
        ctk.CTkLabel(header_frame, text="Status", font=ctk.CTkFont(weight="bold")).pack(
            side="right", padx=10
        )

        # Scrollable model list
        self.model_list_scrollable = ctk.CTkScrollableFrame(
            self.model_list_frame, height=200
        )
        self.model_list_scrollable.pack(fill="both", expand=True, padx=5, pady=(0, 10))

    def setup_model_info_section(self):
        """Setup model information section."""
        # Model Info Frame
        self.model_info_frame = ctk.CTkFrame(self.main_frame)
        self.model_info_frame.pack(fill="x", padx=10, pady=(0, 15))

        # Section title
        section_title = ctk.CTkLabel(
            self.model_info_frame,
            text="Model Information",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        section_title.pack(pady=(15, 10))

        # Info content frame
        self.info_content_frame = ctk.CTkFrame(self.model_info_frame)
        self.info_content_frame.pack(fill="x", padx=15, pady=(0, 15))

        # Model info labels
        self.model_name_label = ctk.CTkLabel(
            self.info_content_frame, text="No model selected"
        )
        self.model_name_label.pack(anchor="w", padx=10, pady=5)

        self.model_details_label = ctk.CTkLabel(
            self.info_content_frame, text="", justify="left"
        )
        self.model_details_label.pack(anchor="w", padx=10, pady=5)

    def setup_embedding_config_section(self):
        """Setup embedding configuration section."""
        # Embedding Config Frame
        self.config_frame = ctk.CTkFrame(self.main_frame)
        self.config_frame.pack(fill="x", padx=10, pady=(0, 15))

        # Section title
        section_title = ctk.CTkLabel(
            self.config_frame,
            text="Embedding Configuration",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        section_title.pack(pady=(15, 10))

        # Config content frame
        config_content_frame = ctk.CTkFrame(self.config_frame)
        config_content_frame.pack(fill="x", padx=15, pady=(0, 15))

        # Batch size
        batch_frame = ctk.CTkFrame(config_content_frame)
        batch_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(batch_frame, text="Batch Size:").pack(
            side="left", padx=10, pady=10
        )
        self.batch_size_entry = ctk.CTkEntry(
            batch_frame, width=100, placeholder_text="32"
        )
        self.batch_size_entry.pack(side="right", padx=10, pady=10)
        self.batch_size_entry.insert(0, "32")

        # Max length
        length_frame = ctk.CTkFrame(config_content_frame)
        length_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(length_frame, text="Max Length:").pack(
            side="left", padx=10, pady=10
        )
        self.max_length_entry = ctk.CTkEntry(
            length_frame, width=100, placeholder_text="Auto"
        )
        self.max_length_entry.pack(side="right", padx=10, pady=10)

        # Normalization
        norm_frame = ctk.CTkFrame(config_content_frame)
        norm_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(norm_frame, text="Normalize Embeddings:").pack(
            side="left", padx=10, pady=10
        )
        self.normalize_checkbox = ctk.CTkCheckBox(
            norm_frame, text="", onvalue=True, offvalue=False
        )
        self.normalize_checkbox.pack(side="right", padx=10, pady=10)
        self.normalize_checkbox.select()  # Default to True

        # Device selection
        device_frame = ctk.CTkFrame(config_content_frame)
        device_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(device_frame, text="Device:").pack(side="left", padx=10, pady=10)
        self.device_menu = ctk.CTkOptionMenu(
            device_frame, values=["auto", "cpu", "cuda"]
        )
        self.device_menu.pack(side="right", padx=10, pady=10)
        self.device_menu.set("auto")

    def setup_cache_management_section(self):
        """Setup cache management section."""
        # Cache Management Frame
        self.cache_frame = ctk.CTkFrame(self.main_frame)
        self.cache_frame.pack(fill="x", padx=10, pady=(0, 15))

        # Section title
        section_title = ctk.CTkLabel(
            self.cache_frame,
            text="Cache Management",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        section_title.pack(pady=(15, 10))

        # Cache stats frame
        cache_stats_frame = ctk.CTkFrame(self.cache_frame)
        cache_stats_frame.pack(fill="x", padx=15, pady=(0, 10))

        self.cache_stats_label = ctk.CTkLabel(
            cache_stats_frame, text="Loading cache statistics...", justify="left"
        )
        self.cache_stats_label.pack(anchor="w", padx=10, pady=10)

        # Cache controls frame
        cache_controls_frame = ctk.CTkFrame(self.cache_frame)
        cache_controls_frame.pack(fill="x", padx=15, pady=(0, 15))

        self.refresh_cache_button = ctk.CTkButton(
            cache_controls_frame,
            text="🔄 Refresh Stats",
            command=self.refresh_cache_stats,
            width=120,
        )
        self.refresh_cache_button.pack(side="left", padx=5, pady=10)

        self.clear_cache_button = ctk.CTkButton(
            cache_controls_frame,
            text="🗑️ Clear Cache",
            command=self.clear_cache,
            width=120,
            fg_color="red",
            hover_color="darkred",
        )
        self.clear_cache_button.pack(side="left", padx=5, pady=10)

        self.cleanup_cache_button = ctk.CTkButton(
            cache_controls_frame,
            text="🧹 Cleanup Old",
            command=self.cleanup_cache,
            width=120,
        )
        self.cleanup_cache_button.pack(side="left", padx=5, pady=10)

    def setup_actions_section(self):
        """Setup actions section."""
        # Actions Frame
        self.actions_frame = ctk.CTkFrame(self.main_frame)
        self.actions_frame.pack(fill="x", padx=10, pady=(0, 15))

        # Section title
        section_title = ctk.CTkLabel(
            self.actions_frame, text="Actions", font=ctk.CTkFont(size=16, weight="bold")
        )
        section_title.pack(pady=(15, 10))

        # Actions controls frame
        actions_controls_frame = ctk.CTkFrame(self.actions_frame)
        actions_controls_frame.pack(fill="x", padx=15, pady=(0, 15))

        self.test_model_button = ctk.CTkButton(
            actions_controls_frame,
            text="🧪 Test Model",
            command=self.test_selected_model,
            width=140,
            state="disabled",
        )
        self.test_model_button.pack(side="left", padx=5, pady=10)

        self.load_model_button = ctk.CTkButton(
            actions_controls_frame,
            text="📥 Load Model",
            command=self.load_selected_model,
            width=140,
            state="disabled",
        )
        self.load_model_button.pack(side="left", padx=5, pady=10)

        self.unload_model_button = ctk.CTkButton(
            actions_controls_frame,
            text="📤 Unload Model",
            command=self.unload_selected_model,
            width=140,
            state="disabled",
        )
        self.unload_model_button.pack(side="left", padx=5, pady=10)

        self.generate_embeddings_button = ctk.CTkButton(
            actions_controls_frame,
            text="🚀 Generate Embeddings",
            command=self.generate_embeddings,
            width=160,
            state="disabled",
        )
        self.generate_embeddings_button.pack(side="left", padx=5, pady=10)

    def load_models(self):
        """Load available models."""
        self.loading_label.configure(text="Loading models...")

        def load_task():
            try:
                # Discover local models
                local_models = self.embedding_controller.discover_local_models()

                # Get all available models (local + popular)
                all_models = self.embedding_controller.get_available_models()

                # Update UI on main thread
                self.parent.after(0, self._update_model_list, all_models)

            except Exception as e:
                logger.error(f"Error loading models: {str(e)}")
                self.parent.after(0, self._show_model_error, str(e))

        threading.Thread(target=load_task, daemon=True).start()

    def refresh_models(self):
        """Refresh the model list."""
        self.load_models()

    def _update_model_list(self, models: List[ModelInfo]):
        """Update the model list UI."""
        self.available_models = models
        self.loading_label.configure(text="")

        # Clear existing model list
        for widget in self.model_list_scrollable.winfo_children():
            widget.destroy()

        # Add models to list
        for model in models:
            self._create_model_item(model)

        # Update cache stats
        self.refresh_cache_stats()

        logger.info(f"Loaded {len(models)} models")

    def _create_model_item(self, model: ModelInfo):
        """Create a model item in the list."""
        item_frame = ctk.CTkFrame(self.model_list_scrollable)
        item_frame.pack(fill="x", padx=5, pady=2)

        # Model info
        info_frame = ctk.CTkFrame(item_frame)
        info_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        # Model name
        name_label = ctk.CTkLabel(
            info_frame,
            text=model.display_name,
            font=ctk.CTkFont(weight="bold"),
            anchor="w",
        )
        name_label.pack(anchor="w", padx=10, pady=(5, 0))

        # Model details
        details = []
        if model.embedding_dimension:
            details.append(f"Dim: {model.embedding_dimension}")
        if model.model_size_mb:
            details.append(f"Size: {model.model_size_mb:.1f}MB")
        if model.supports_languages and model.supports_languages != ["en"]:
            details.append(f"Lang: {', '.join(model.supports_languages)}")

        if details:
            details_label = ctk.CTkLabel(
                info_frame,
                text=" | ".join(details),
                font=ctk.CTkFont(size=11),
                text_color="gray",
                anchor="w",
            )
            details_label.pack(anchor="w", padx=10, pady=(0, 5))

        # Status and select button
        controls_frame = ctk.CTkFrame(item_frame)
        controls_frame.pack(side="right", padx=5, pady=5)

        # Status indicator
        status_color = (
            "green" if model.is_loaded else ("orange" if model.is_local else "red")
        )
        status_text = (
            "Loaded"
            if model.is_loaded
            else ("Available" if model.is_local else "Not Downloaded")
        )

        status_label = ctk.CTkLabel(
            controls_frame,
            text=status_text,
            text_color=status_color,
            font=ctk.CTkFont(size=11, weight="bold"),
        )
        status_label.pack(padx=10, pady=(5, 0))

        # Select button
        select_button = ctk.CTkButton(
            controls_frame,
            text="Select",
            command=lambda m=model: self.select_model(m),
            width=80,
            height=25,
        )
        select_button.pack(padx=10, pady=(0, 5))

    def select_model(self, model: ModelInfo):
        """Select a model for configuration."""
        try:
            logger.debug(f"Selecting model: {model.model_name} (is_loaded: {model.is_loaded}, is_local: {model.is_local})")
            self.selected_model = model
            self._update_model_info()
            self._update_action_buttons()
            logger.info(f"Selected model: {model.model_name}")
        except Exception as e:
            logger.error(f"Error selecting model: {e}")
            self.selected_model = None
            self._update_model_info()
            self._update_action_buttons()

    def _update_model_info(self):
        """Update the model information display."""
        if not self.selected_model:
            self.model_name_label.configure(text="No model selected")
            self.model_details_label.configure(text="")
            return

        model = self.selected_model

        # Model name and status
        status = (
            "✅ Loaded"
            if model.is_loaded
            else ("📦 Available locally" if model.is_local else "⬇️ Download required")
        )
        self.model_name_label.configure(text=f"{model.display_name} - {status}")

        # Model details
        details = []
        if model.description:
            details.append(f"Description: {model.description}")
        if model.embedding_dimension:
            details.append(f"Embedding Dimension: {model.embedding_dimension}")
        if model.max_sequence_length:
            details.append(f"Max Sequence Length: {model.max_sequence_length}")
        if model.model_size_mb:
            details.append(f"Model Size: {model.model_size_mb:.1f} MB")
        if model.memory_usage_mb:
            details.append(f"Memory Usage: {model.memory_usage_mb:.1f} MB")
        if model.load_time_seconds:
            details.append(f"Load Time: {model.load_time_seconds:.2f}s")

        self.model_details_label.configure(text="\n".join(details))

    def _update_action_buttons(self):
        """Update the state of action buttons."""
        try:
            if not self.selected_model:
                logger.debug("No model selected - disabling all buttons")
                self._set_all_buttons_disabled()
                return

            model = self.selected_model
            logger.debug(f"Updating buttons for model: {model.model_name}, is_loaded: {model.is_loaded}, is_local: {model.is_local}")

            if model.is_loaded:
                logger.debug("Model is loaded - enabling test, unload, and generate buttons")
                self.test_model_button.configure(state="normal")
                self.load_model_button.configure(state="disabled")
                self.unload_model_button.configure(state="normal")
                self.generate_embeddings_button.configure(state="normal")
            elif model.is_local:
                logger.debug("Model is local but not loaded - enabling load button")
                self.test_model_button.configure(state="disabled")
                self.load_model_button.configure(state="normal")
                self.unload_model_button.configure(state="disabled")
                self.generate_embeddings_button.configure(state="disabled")
            else:
                logger.debug("Model is not local - disabling all action buttons")
                self._set_all_buttons_disabled()
                
        except Exception as e:
            logger.error(f"Error updating action buttons: {e}")
            self._set_all_buttons_disabled()
    
    def _set_all_buttons_disabled(self):
        """Helper method to disable all action buttons."""
        try:
            self.test_model_button.configure(state="disabled")
            self.load_model_button.configure(state="disabled")
            self.unload_model_button.configure(state="disabled")
            self.generate_embeddings_button.configure(state="disabled")
        except Exception as e:
            logger.error(f"Error disabling buttons: {e}")

    def load_selected_model(self):
        """Load the selected model."""
        if not self.selected_model:
            messagebox.showwarning("No Model", "Please select a model first.")
            return

        self.load_model_button.configure(state="disabled", text="Loading...")

        def load_task():
            try:
                success = self.embedding_controller.load_model(self.selected_model)

                def update_ui():
                    self.load_model_button.configure(
                        state="normal", text="📥 Load Model"
                    )
                    if success:
                        self.selected_model.is_loaded = True
                        self._update_model_info()
                        self._update_action_buttons()
                        # Update the model list to reflect new status
                        for model in self.available_models:
                            if model.model_name == self.selected_model.model_name:
                                model.is_loaded = True
                                break
                        # Note: Don't recreate the entire model list UI as it interferes with buttons
                        # The action buttons are already updated by _update_action_buttons() above
                        messagebox.showinfo(
                            "Success",
                            f"Model '{self.selected_model.display_name}' loaded successfully!",
                        )
                    else:
                        messagebox.showerror(
                            "Error",
                            f"Failed to load model '{self.selected_model.display_name}'",
                        )

                self.parent.after(0, update_ui)

            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                self.parent.after(
                    0,
                    lambda: messagebox.showerror(
                        "Error", f"Error loading model: {str(e)}"
                    ),
                )
                self.parent.after(
                    0,
                    lambda: self.load_model_button.configure(
                        state="normal", text="📥 Load Model"
                    ),
                )

        threading.Thread(target=load_task, daemon=True).start()

    def unload_selected_model(self):
        """Unload the selected model."""
        if not self.selected_model or not self.selected_model.is_loaded:
            messagebox.showwarning(
                "No Loaded Model", "Please select a loaded model first."
            )
            return

        try:
            success = self.embedding_controller.unload_model(
                self.selected_model.model_name
            )

            if success:
                self.selected_model.is_loaded = False
                self._update_model_info()
                self._update_action_buttons()
                # Update the model list to reflect new status
                for model in self.available_models:
                    if model.model_name == self.selected_model.model_name:
                        model.is_loaded = False
                        break
                self._update_model_list(self.available_models)
                messagebox.showinfo(
                    "Success",
                    f"Model '{self.selected_model.display_name}' unloaded successfully!",
                )
            else:
                messagebox.showerror(
                    "Error",
                    f"Failed to unload model '{self.selected_model.display_name}'",
                )

        except Exception as e:
            logger.error(f"Error unloading model: {str(e)}")
            messagebox.showerror("Error", f"Error unloading model: {str(e)}")

    def test_selected_model(self):
        """Test the performance of the selected model."""
        if not self.selected_model or not self.selected_model.is_loaded:
            messagebox.showwarning(
                "No Loaded Model", "Please select and load a model first."
            )
            return

        # Sample texts for testing
        sample_texts = [
            "This is a sample document for testing embedding generation.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning and artificial intelligence are transforming many industries.",
            "Natural language processing enables computers to understand human language.",
            "Text embedding models convert text into numerical representations.",
        ]

        self.test_model_button.configure(state="disabled", text="Testing...")

        def test_task():
            try:
                results = self.embedding_controller.test_model_performance(
                    self.selected_model, sample_texts
                )

                def show_results():
                    self.test_model_button.configure(
                        state="normal", text="🧪 Test Model"
                    )

                    if "error" in results:
                        messagebox.showerror("Test Error", results["error"])
                    else:
                        # Format results
                        result_text = f"""Model Performance Test Results:

Model: {self.selected_model.display_name}

Processing Time: {results.get('processing_time_seconds', 0):.3f} seconds
Processing Speed: {results.get('texts_per_second', 0):.1f} texts/second
Embedding Dimension: {results.get('embedding_dimension', 'Unknown')}
Memory Usage: {results.get('memory_usage_mb', 0):.1f} MB

Sample texts processed: {len(sample_texts)}"""

                        messagebox.showinfo("Performance Test Results", result_text)

                self.parent.after(0, show_results)

            except Exception as e:
                logger.error(f"Error testing model: {str(e)}")
                self.parent.after(
                    0,
                    lambda: messagebox.showerror(
                        "Error", f"Error testing model: {str(e)}"
                    ),
                )
                self.parent.after(
                    0,
                    lambda: self.test_model_button.configure(
                        state="normal", text="🧪 Test Model"
                    ),
                )

        threading.Thread(target=test_task, daemon=True).start()

    def generate_embeddings(self):
        """Generate embeddings for the loaded data using the selected model."""
        if not self.selected_model or not self.selected_model.is_loaded:
            messagebox.showwarning(
                "No Loaded Model", "Please select and load a model first."
            )
            return

        # Get data from main window
        main_window = self._get_main_window()
        if not main_window:
            messagebox.showerror("Error", "Cannot access main window")
            return

        # Get data configuration
        data_config = main_window.get_data_config()
        if not data_config or not data_config.is_configured:
            messagebox.showerror(
                "No Data", "Please load and configure data in the Data Import tab first"
            )
            return

        # Get embedding configuration
        embedding_config = self.get_embedding_config()

        # Update button state
        self.generate_embeddings_button.configure(
            state="disabled", text="Generating..."
        )

        def embedding_task():
            try:
                # Generate embeddings
                success = self.embedding_controller.generate_embeddings(
                    data_config=data_config,
                    embedding_config=embedding_config,
                    progress_callback=self._on_embedding_progress,
                    completion_callback=self._on_embedding_complete,
                    error_callback=self._on_embedding_error,
                )

                if not success:

                    def restore_button():
                        self.generate_embeddings_button.configure(
                            state="normal", text="🚀 Generate Embeddings"
                        )
                        messagebox.showerror(
                            "Error", "Failed to start embedding generation"
                        )

                    self.parent.after(0, restore_button)

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error starting embedding generation: {error_msg}")

                def show_error():
                    self.generate_embeddings_button.configure(
                        state="normal", text="🚀 Generate Embeddings"
                    )
                    messagebox.showerror(
                        "Error", f"Error starting embedding generation: {error_msg}"
                    )

                self.parent.after(0, show_error)

        threading.Thread(target=embedding_task, daemon=True).start()

    def _get_main_window(self):
        """Get reference to main window."""
        # Navigate up the widget hierarchy to find main window
        widget = self.parent
        while widget:
            if hasattr(widget, "get_data_config"):
                return widget
            widget = widget.master
        return None

    def _on_embedding_progress(self, current: int, total: int, message: str):
        """Handle embedding progress updates."""

        def update_ui():
            progress_text = f"Generating... {current}/{total} - {message}"
            self.generate_embeddings_button.configure(text=progress_text)

        self.parent.after(0, update_ui)

    def _on_embedding_complete(self, result: EmbeddingResult):
        """Handle embedding completion."""

        def update_ui():
            self.current_embedding_result = result
            self.generate_embeddings_button.configure(
                state="normal", text="🚀 Generate Embeddings"
            )
            messagebox.showinfo(
                "Success",
                f"Embeddings generated successfully!\n\n"
                f"Documents: {len(result.texts):,}\n"
                f"Embedding dimension: {result.embeddings.shape[1] if result.embeddings is not None else 'Unknown'}\n"
                f"Processing time: {getattr(result, 'processing_time_seconds', 0):.2f}s",
            )

        self.parent.after(0, update_ui)

    def _on_embedding_error(self, error_message: str):
        """Handle embedding errors."""

        def update_ui():
            self.generate_embeddings_button.configure(
                state="normal", text="🚀 Generate Embeddings"
            )
            messagebox.showerror("Embedding Error", error_message)

        self.parent.after(0, update_ui)

    def refresh_cache_stats(self):
        """Refresh cache statistics."""
        try:
            stats = self.embedding_controller.get_cache_stats()

            if "error" in stats:
                self.cache_stats_label.configure(
                    text=f"Error loading cache stats: {stats['error']}"
                )
            else:
                stats_text = f"""Cache Statistics:
                
Total Files: {stats.get('total_files', 0)}
Total Size: {stats.get('total_size_mb', 0):.1f} MB ({stats.get('total_size_gb', 0):.2f} GB)
Usage: {stats.get('usage_percentage', 0):.1f}% of {stats.get('max_size_gb', 0):.1f} GB limit

Cache Directory: {stats.get('cache_dir', 'Unknown')}"""

                self.cache_stats_label.configure(text=stats_text)

        except Exception as e:
            logger.error(f"Error refreshing cache stats: {str(e)}")
            self.cache_stats_label.configure(
                text=f"Error loading cache stats: {str(e)}"
            )

    def clear_cache(self):
        """Clear the embedding cache."""
        result = messagebox.askyesno(
            "Clear Cache",
            "Are you sure you want to clear all cached embeddings?\n\nThis will permanently delete all cached data and cannot be undone.",
        )

        if result:
            try:
                success = self.embedding_controller.clear_cache()
                if success:
                    messagebox.showinfo("Success", "Cache cleared successfully!")
                    self.refresh_cache_stats()
                else:
                    messagebox.showerror("Error", "Failed to clear cache.")
            except Exception as e:
                logger.error(f"Error clearing cache: {str(e)}")
                messagebox.showerror("Error", f"Error clearing cache: {str(e)}")

    def cleanup_cache(self):
        """Clean up old cache entries."""
        try:
            removed_count = self.embedding_controller.cleanup_expired_cache(
                max_age_days=30
            )
            messagebox.showinfo(
                "Cleanup Complete", f"Cleaned up {removed_count} expired cache entries."
            )
            self.refresh_cache_stats()
        except Exception as e:
            logger.error(f"Error cleaning up cache: {str(e)}")
            messagebox.showerror("Error", f"Error cleaning up cache: {str(e)}")

    def get_embedding_config(self) -> EmbeddingConfig:
        """Get the current embedding configuration."""
        config = EmbeddingConfig()
        config.model_info = self.selected_model

        try:
            config.batch_size = int(self.batch_size_entry.get() or "32")
        except ValueError:
            config.batch_size = 32

        max_length_text = self.max_length_entry.get().strip()
        if max_length_text and max_length_text.lower() != "auto":
            try:
                config.max_length = int(max_length_text)
            except ValueError:
                config.max_length = None

        config.normalize_embeddings = self.normalize_checkbox.get()
        config.device = self.device_menu.get()

        return config

    def _show_model_error(self, error_message: str):
        """Show model loading error."""
        self.loading_label.configure(text="Error loading models")
        messagebox.showerror(
            "Model Loading Error", f"Failed to load models:\n{error_message}"
        )

    def is_configured(self) -> bool:
        """Check if model configuration is complete."""
        return self.selected_model is not None and self.selected_model.is_loaded
