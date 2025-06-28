"""
Main application window with tabbed interface.
"""

import logging
import customtkinter as ctk
from tkinter import messagebox
from typing import Optional

from ..utils.constants import (
    APP_NAME,
    APP_VERSION,
    WINDOW_MIN_WIDTH,
    WINDOW_MIN_HEIGHT,
    WINDOW_DEFAULT_WIDTH,
    WINDOW_DEFAULT_HEIGHT,
)
from ..utils.error_handling import handle_exception
from ..controllers import EmbeddingController, TopicModelingController
from ..services import (
    FileIOService,
    DataValidationService,
    ModelManagementService,
    CacheService,
    EmbeddingService,
    BERTopicService,
)
from .tabs.data_import_tab import DataImportTab
from .tabs.model_config_tab import ModelConfigTab
from .tabs.topic_modeling_tab import TopicModelingTab
from .tabs.visualization_tab import VisualizationTab
from .tabs.hyperparameter_tab import HyperparameterTab
from .tabs.export_tab import ExportTab
from .dialogs.about_dialog import AboutDialog
from .dialogs.settings_dialog import SettingsDialog

logger = logging.getLogger(__name__)


class MainWindow(ctk.CTk):
    """Main application window with tabbed interface."""

    def __init__(self):
        super().__init__()

        logger.info("Initializing main window")

        # Configure window
        self.title(f"{APP_NAME} v{APP_VERSION}")
        self.geometry(f"{WINDOW_DEFAULT_WIDTH}x{WINDOW_DEFAULT_HEIGHT}")
        self.minsize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        # Center window on screen
        self._center_window()

        # Initialize services and controllers
        self._initialize_services()
        self._initialize_controllers()

        # Create UI components
        self._create_menu()
        self._create_main_content()
        self._create_status_bar()

        # Setup controller integration
        self._setup_controller_integration()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        logger.info("Main window initialized successfully")

    def _initialize_services(self):
        """Initialize all application services."""
        logger.info("Initializing services...")

        try:
            # Core services
            self.file_io_service = FileIOService()
            self.data_validation_service = DataValidationService()
            self.model_management_service = ModelManagementService()
            self.cache_service = CacheService()
            self.embedding_service = EmbeddingService()
            self.bertopic_service = BERTopicService()

            logger.info("All services initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize services: {str(e)}"
            logger.error(error_msg, exc_info=True)
            messagebox.showerror("Initialization Error", error_msg)
            raise

    def _initialize_controllers(self):
        """Initialize all application controllers."""
        logger.info("Initializing controllers...")

        try:
            # Data controller is created by the data import tab

            # Embedding controller
            self.embedding_controller = EmbeddingController()

            # Topic modeling controller
            self.topic_modeling_controller = TopicModelingController()
            self.topic_modeling_controller.set_services(
                embedding_service=self.embedding_service,
                data_validation_service=self.data_validation_service,
            )

            logger.info("All controllers initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize controllers: {str(e)}"
            logger.error(error_msg, exc_info=True)
            messagebox.showerror("Initialization Error", error_msg)
            raise

    def _center_window(self):
        """Center the window on the screen."""
        # Get screen dimensions
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Calculate position
        x = (screen_width - WINDOW_DEFAULT_WIDTH) // 2
        y = (screen_height - WINDOW_DEFAULT_HEIGHT) // 2

        self.geometry(f"{WINDOW_DEFAULT_WIDTH}x{WINDOW_DEFAULT_HEIGHT}+{x}+{y}")

    def _create_menu(self):
        """Create the application menu bar."""
        # Note: CustomTkinter doesn't have native menu support
        # We'll create a custom menu frame
        self.menu_frame = ctk.CTkFrame(self, height=40)
        self.menu_frame.pack(fill="x", padx=5, pady=(5, 0))
        self.menu_frame.pack_propagate(False)

        # File menu button
        self.file_menu_btn = ctk.CTkButton(
            self.menu_frame, text="File", width=60, command=self._show_file_menu
        )
        self.file_menu_btn.pack(side="left", padx=(10, 5), pady=5)

        # Tools menu button
        self.tools_menu_btn = ctk.CTkButton(
            self.menu_frame, text="Tools", width=60, command=self._show_tools_menu
        )
        self.tools_menu_btn.pack(side="left", padx=5, pady=5)

        # Help menu button
        self.help_menu_btn = ctk.CTkButton(
            self.menu_frame, text="Help", width=60, command=self._show_help_menu
        )
        self.help_menu_btn.pack(side="left", padx=5, pady=5)

    def _create_main_content(self):
        """Create the main content area with tabs."""
        # Create tab view
        self.tab_view = ctk.CTkTabview(self, width=WINDOW_DEFAULT_WIDTH - 20)
        self.tab_view.pack(fill="both", expand=True, padx=10, pady=10)

        # Add tabs
        self.tab_view.add("Data Import")
        self.tab_view.add("Model Config")
        self.tab_view.add("Topic Modeling")
        self.tab_view.add("Visualization")
        self.tab_view.add("Hyperparameter")
        self.tab_view.add("Export")

        # Initialize tab content
        try:
            self.data_import_tab = DataImportTab(self.tab_view.tab("Data Import"))
            self.model_config_tab = ModelConfigTab(self.tab_view.tab("Model Config"))
            self.topic_modeling_tab = TopicModelingTab(
                self.tab_view.tab("Topic Modeling")
            )
            self.visualization_tab = VisualizationTab(
                self.tab_view.tab("Visualization")
            )
            self.hyperparameter_tab = HyperparameterTab(
                self.tab_view.tab("Hyperparameter")
            )
            self.export_tab = ExportTab(self.tab_view.tab("Export"))

            # Set default tab
            self.tab_view.set("Data Import")

        except Exception as e:
            error_msg = handle_exception(e, "Failed to initialize tabs")
            messagebox.showerror("Initialization Error", error_msg)
            logger.error(f"Tab initialization failed: {e}")

    def _setup_controller_integration(self):
        """Setup integration between controllers and tabs."""
        logger.info("Setting up controller integration...")

        # Data import tab creates its own controller internally
        # Model config tab creates its own controller internally

        # Connect topic modeling controller to topic modeling tab
        try:
            self.topic_modeling_tab.set_controller(self.topic_modeling_controller)
            logger.info("Topic modeling controller connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect topic modeling controller: {str(e)}")

        # Setup main window status callbacks for all controllers
        try:
            self._setup_status_callbacks()
            logger.info("Status callbacks setup complete")
        except Exception as e:
            logger.error(f"Failed to setup status callbacks: {str(e)}")

        logger.info("Controller integration setup complete")

    def _setup_status_callbacks(self):
        """Setup status callbacks for all controllers."""
        # Data controller callbacks are handled by the data import tab

        # Embedding controller callbacks are set when calling generate_embeddings

        # Topic modeling controller callbacks
        self.topic_modeling_controller.set_callbacks(
            progress_callback=self._update_main_progress,
            status_callback=self.set_status,
            error_callback=self._show_error_message,
        )

    def _update_main_progress(self, progress_info):
        """Update main window progress display."""
        if hasattr(progress_info, "progress_percentage"):
            self.show_progress(progress_info.progress_percentage / 100.0)
        if hasattr(progress_info, "current_step"):
            self.set_status(progress_info.current_step)

    def _show_error_message(self, error_message: str):
        """Show error message in a dialog."""
        messagebox.showerror("Error", error_message)
        self.set_status(f"Error: {error_message}")

    def _create_status_bar(self):
        """Create the status bar."""
        self.status_frame = ctk.CTkFrame(self, height=30)
        self.status_frame.pack(fill="x", padx=5, pady=(0, 5))
        self.status_frame.pack_propagate(False)

        # Status label
        self.status_label = ctk.CTkLabel(self.status_frame, text="Ready", anchor="w")
        self.status_label.pack(side="left", padx=10, pady=5)

        # Progress bar (initially hidden)
        self.progress_bar = ctk.CTkProgressBar(self.status_frame)
        self.progress_bar.pack(side="right", padx=10, pady=5)
        self.progress_bar.pack_forget()  # Hide initially

    def _show_file_menu(self):
        """Show file menu options."""
        # Create a simple popup menu simulation
        menu_window = ctk.CTkToplevel(self)
        menu_window.title("File Menu")
        menu_window.geometry("200x150")

        # Menu options
        new_btn = ctk.CTkButton(
            menu_window, text="New Project", command=self._new_project
        )
        new_btn.pack(pady=5)

        open_btn = ctk.CTkButton(
            menu_window, text="Open Project", command=self._open_project
        )
        open_btn.pack(pady=5)

        save_btn = ctk.CTkButton(
            menu_window, text="Save Project", command=self._save_project
        )
        save_btn.pack(pady=5)

        exit_btn = ctk.CTkButton(menu_window, text="Exit", command=self._on_closing)
        exit_btn.pack(pady=5)

    def _show_tools_menu(self):
        """Show tools menu options."""
        menu_window = ctk.CTkToplevel(self)
        menu_window.title("Tools Menu")
        menu_window.geometry("200x100")

        settings_btn = ctk.CTkButton(
            menu_window, text="Settings", command=self._show_settings
        )
        settings_btn.pack(pady=10)

        cache_btn = ctk.CTkButton(
            menu_window, text="Clear Cache", command=self._clear_cache
        )
        cache_btn.pack(pady=10)

    def _show_help_menu(self):
        """Show help menu options."""
        menu_window = ctk.CTkToplevel(self)
        menu_window.title("Help Menu")
        menu_window.geometry("200x100")

        about_btn = ctk.CTkButton(menu_window, text="About", command=self._show_about)
        about_btn.pack(pady=10)

        help_btn = ctk.CTkButton(
            menu_window, text="User Manual", command=self._show_help
        )
        help_btn.pack(pady=10)

    def set_status(self, message: str):
        """Update the status bar message."""
        self.status_label.configure(text=message)
        logger.info(f"Status: {message}")

    def show_progress(self, progress: Optional[float] = None):
        """Show progress bar with optional progress value."""
        self.progress_bar.pack(side="right", padx=10, pady=5)
        if progress is not None:
            self.progress_bar.set(progress)

    def hide_progress(self):
        """Hide progress bar."""
        self.progress_bar.pack_forget()

    def get_data_config(self):
        """Get current data configuration from data import tab."""
        return self.data_import_tab.get_data_controller().data_config

    def get_embedding_result(self):
        """Get current embedding result from model config tab."""
        # Model config tab handles embedding generation
        return getattr(self.model_config_tab, "current_embedding_result", None)

    def get_topic_modeling_result(self):
        """Get current topic modeling result."""
        return self.topic_modeling_controller.get_current_result()

    def _new_project(self):
        """Create a new project."""
        self.set_status("New project created")
        # Clear all controllers
        self.data_import_tab.get_data_controller().clear_data()
        # Embedding controller doesn't store results to clear
        self.topic_modeling_controller.clear_current_results()

    def _open_project(self):
        """Open an existing project."""
        self.set_status("Opening project...")
        # TODO: Implement open project logic

    def _save_project(self):
        """Save current project."""
        self.set_status("Saving project...")
        # TODO: Implement save project logic

    def _show_settings(self):
        """Show settings dialog."""
        try:
            dialog = SettingsDialog(self)
            dialog.focus()
        except Exception as e:
            error_msg = handle_exception(e, "Failed to open settings")
            messagebox.showerror("Error", error_msg)

    def _show_about(self):
        """Show about dialog."""
        try:
            dialog = AboutDialog(self)
            dialog.focus()
        except Exception as e:
            error_msg = handle_exception(e, "Failed to open about dialog")
            messagebox.showerror("Error", error_msg)

    def _show_help(self):
        """Show help documentation."""
        self.set_status("Opening help documentation...")
        # TODO: Implement help system
        messagebox.showinfo("Help", "Help system not yet implemented.")

    def _clear_cache(self):
        """Clear application cache."""
        result = messagebox.askyesno(
            "Clear Cache", "Are you sure you want to clear all cached data?"
        )
        if result:
            self.set_status("Clearing cache...")
            try:
                self.cache_service.clear_all_cache()
                messagebox.showinfo("Cache Cleared", "Cache cleared successfully.")
                self.set_status("Cache cleared")
            except Exception as e:
                error_msg = f"Failed to clear cache: {str(e)}"
                messagebox.showerror("Error", error_msg)
                self.set_status(error_msg)

    def _on_closing(self):
        """Handle window closing event."""
        result = messagebox.askyesno(
            "Exit Application", "Are you sure you want to exit?"
        )
        if result:
            logger.info("Application closing - cleaning up resources")

            # Cleanup controllers
            try:
                # EmbeddingController doesn't have cleanup method
                self.topic_modeling_controller.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

            logger.info("Application closing")
            self.destroy()
