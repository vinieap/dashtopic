"""
Data Import Tab - Enhanced file loading and column selection interface for Phase 2.
"""
import logging
import customtkinter as ctk
from tkinter import filedialog, messagebox
from typing import Optional

from ...controllers.data_controller import DataController
from ..widgets import DataPreviewWidget, ColumnSelectorWidget, TextPreviewWidget

logger = logging.getLogger(__name__)


class DataImportTab:
    """Enhanced tab for importing and configuring data sources with full Phase 2 functionality."""
    
    def __init__(self, parent):
        self.parent = parent
        self.data_controller = DataController()
        
        # Connect callbacks
        self.data_controller.set_status_callback(self._on_status_update)
        self.data_controller.set_progress_callback(self._on_progress_update)
        
        self.setup_ui()
        logger.info("Enhanced Data Import tab initialized")
    
    def setup_ui(self):
        """Set up the enhanced tab user interface."""
        # Main scrollable frame
        self.main_frame = ctk.CTkScrollableFrame(self.parent)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Bind mouse wheel for main scrolling
        self._bind_mousewheel(self.main_frame)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Data Import & Configuration",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.title_label.pack(pady=(10, 20))
        
        # File selection section
        self._create_file_selection_section()
        
        # Data preview section  
        self._create_data_preview_section()
        
        # Column selection section
        self._create_column_selection_section()
        
        # Text combination preview section
        self._create_text_preview_section()
        
        # Validation and status section
        self._create_validation_section()
        
        # Action buttons section
        self._create_action_buttons_section()
    
    def _create_file_selection_section(self):
        """Create enhanced file selection interface."""
        # File selection frame
        file_frame = ctk.CTkFrame(self.main_frame)
        file_frame.pack(fill="x", padx=10, pady=5)
        
        # Section header
        ctk.CTkLabel(
            file_frame, 
            text="1. Select Dataset File",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        # File path entry and browse button
        path_frame = ctk.CTkFrame(file_frame, fg_color="transparent")
        path_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.file_path_var = ctk.StringVar()
        self.file_path_entry = ctk.CTkEntry(
            path_frame,
            textvariable=self.file_path_var,
            placeholder_text="Select a dataset file (CSV, Excel, Parquet, Feather)...",
            font=ctk.CTkFont(size=12)
        )
        self.file_path_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.browse_button = ctk.CTkButton(
            path_frame,
            text="Browse",
            width=100,
            command=self._browse_file
        )
        self.browse_button.pack(side="right")
        
        # File info frame
        self.file_info_frame = ctk.CTkFrame(file_frame)
        self.file_info_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.file_info_label = ctk.CTkLabel(
            self.file_info_frame,
            text="No file selected",
            anchor="w",
            font=ctk.CTkFont(size=11)
        )
        self.file_info_label.pack(fill="x", padx=10, pady=10)
        
        # Load button
        self.load_button = ctk.CTkButton(
            file_frame,
            text="Load Data",
            width=120,
            command=self._load_data,
            state="disabled"
        )
        self.load_button.pack(anchor="e", padx=10, pady=(0, 10))
    
    def _create_data_preview_section(self):
        """Create data preview section."""
        # Preview frame
        preview_frame = ctk.CTkFrame(self.main_frame)
        preview_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Section header
        ctk.CTkLabel(
            preview_frame, 
            text="2. Data Preview & Statistics",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Data preview widget
        self.data_preview_widget = DataPreviewWidget(preview_frame)
        self.data_preview_widget.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    
    def _create_column_selection_section(self):
        """Create column selection section."""
        # Column selection frame
        column_frame = ctk.CTkFrame(self.main_frame)
        column_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Section header
        ctk.CTkLabel(
            column_frame, 
            text="3. Select Text Columns for Analysis",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Column selector widget
        self.column_selector_widget = ColumnSelectorWidget(column_frame)
        self.column_selector_widget.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Set up callbacks
        self.column_selector_widget.set_selection_callback(self._on_column_selection_changed)
        self.column_selector_widget.set_settings_callback(self._on_combination_settings_changed)
    
    def _create_text_preview_section(self):
        """Create text combination preview section."""
        # Text preview frame
        text_frame = ctk.CTkFrame(self.main_frame)
        text_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Section header
        ctk.CTkLabel(
            text_frame, 
            text="4. Text Combination Preview",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Text preview widget
        self.text_preview_widget = TextPreviewWidget(text_frame)
        self.text_preview_widget.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    
    def _create_validation_section(self):
        """Create validation and issues section."""
        # Validation frame
        validation_frame = ctk.CTkFrame(self.main_frame)
        validation_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Section header
        ctk.CTkLabel(
            validation_frame, 
            text="5. Data Validation & Issues",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Validation display
        self.validation_text = ctk.CTkTextbox(
            validation_frame,
            height=80,
            state="disabled"
        )
        self.validation_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Bind mouse wheel for validation text
        self._bind_validation_mousewheel(self.validation_text)
        
        # Initially show no validation info
        self._update_validation_display([], [])
    
    def _create_action_buttons_section(self):
        """Create action buttons section."""
        # Actions frame
        actions_frame = ctk.CTkFrame(self.main_frame)
        actions_frame.pack(fill="x", padx=10, pady=5)
        
        # Section header
        ctk.CTkLabel(
            actions_frame, 
            text="6. Actions",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Buttons frame
        buttons_frame = ctk.CTkFrame(actions_frame, fg_color="transparent")
        buttons_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Clear data button
        self.clear_button = ctk.CTkButton(
            buttons_frame,
            text="Clear Data",
            width=120,
            command=self._clear_data
        )
        self.clear_button.pack(side="left", padx=(0, 10))
        
        # Ready status
        self.ready_label = ctk.CTkLabel(
            buttons_frame,
            text="Status: No data loaded",
            font=ctk.CTkFont(size=12)
        )
        self.ready_label.pack(side="left", expand=True, fill="x")
        
        # Continue to next step button
        self.continue_button = ctk.CTkButton(
            buttons_frame,
            text="Continue to Model Configuration →",
            width=250,
            command=self._continue_to_next_step,
            state="disabled"
        )
        self.continue_button.pack(side="right")
    
    def _browse_file(self):
        """Open enhanced file browser dialog."""
        filetypes = [
            ("All Supported", "*.csv *.xlsx *.xls *.parquet *.feather"),
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx *.xls"),
            ("Parquet files", "*.parquet"),
            ("Feather files", "*.feather"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Dataset File",
            filetypes=filetypes
        )
        
        if filename:
            self.file_path_var.set(filename)
            self._update_file_info(filename)
            self.load_button.configure(state="normal")
            logger.info(f"File selected: {filename}")
    
    def _update_file_info(self, filename: str):
        """Update file information display with enhanced details."""
        try:
            file_info = self.data_controller.get_file_info(filename)
            
            if file_info['supported']:
                info_text = (f"File: {file_info['file_name']}\n"
                           f"Size: {file_info['file_size_mb']:.1f} MB\n"
                           f"Format: {file_info['file_format'].upper()}")
                text_color = None
            else:
                info_text = (f"File: {file_info['file_name']}\n"
                           f"⚠️ Unsupported format: {file_info['file_format']}")
                text_color = "orange"
                self.load_button.configure(state="disabled")
            
            if text_color:
                self.file_info_label.configure(text=info_text, text_color=text_color)
            else:
                self.file_info_label.configure(text=info_text)
            
        except Exception as e:
            error_text = f"❌ Error reading file: {str(e)}"
            self.file_info_label.configure(text=error_text, text_color="red")
            self.load_button.configure(state="disabled")
            logger.error(f"Error reading file info: {e}")
    
    def _load_data(self):
        """Load and validate the selected data file."""
        if not self.file_path_var.get():
            messagebox.showwarning("No File", "Please select a file first.")
            return
        
        try:
            # Disable load button during loading
            self.load_button.configure(state="disabled", text="Loading...")
            
            # Load file through controller
            success, message = self.data_controller.load_file(self.file_path_var.get())
            
            if success:
                # Update UI with loaded data
                self._update_ui_after_loading()
                
                # Show success message
                messagebox.showinfo("Success", message)
                
            else:
                # Show error message
                messagebox.showerror("Load Error", message)
                
        except Exception as e:
            error_msg = f"Unexpected error during loading: {str(e)}"
            messagebox.showerror("Error", error_msg)
            logger.error(f"Data loading exception: {e}")
            
        finally:
            # Re-enable load button
            self.load_button.configure(state="normal", text="Load Data")
    
    def _update_ui_after_loading(self):
        """Update all UI components after successful data loading."""
        # Get data and summary from controller
        preview_data = self.data_controller.get_data_preview(max_rows=50)
        summary = self.data_controller.get_data_summary()
        column_analysis = self.data_controller.get_column_analysis()
        recommended_columns = self.data_controller.get_recommended_columns()
        errors, warnings = self.data_controller.get_validation_issues()
        
        # Update data preview
        self.data_preview_widget.update_data(preview_data, summary)
        
        # Update column selector
        if self.data_controller.current_metadata:
            self.column_selector_widget.set_columns(
                self.data_controller.current_metadata.columns,
                column_analysis,
                recommended_columns
            )
        
        # Update validation display
        self._update_validation_display(errors, warnings)
        
        # Update ready status
        self._update_ready_status()
        
        logger.info("UI updated after successful data loading")
    
    def _on_column_selection_changed(self, selected_columns):
        """Handle column selection changes."""
        # Update controller with new selection
        success, message = self.data_controller.update_column_selection(selected_columns)
        
        if not success:
            messagebox.showwarning("Selection Error", message)
        
        # Update text preview
        self._update_text_preview()
        
        # Update ready status
        self._update_ready_status()
    
    def _on_combination_settings_changed(self):
        """Handle text combination settings changes."""
        settings = self.column_selector_widget.get_combination_settings()
        
        # Update controller settings
        self.data_controller.update_text_combination_settings(
            separator=settings['separator'],
            include_column_names=settings['include_column_names']
        )
        
        # Update text preview
        self._update_text_preview()
    
    def _update_text_preview(self):
        """Update the text combination preview."""
        preview_text = self.data_controller.get_combined_text_preview()
        selected_columns = self.column_selector_widget.get_selected_columns()
        
        if selected_columns:
            status = f"{len(selected_columns)} columns selected"
        else:
            status = "No columns selected"
        
        self.text_preview_widget.update_preview(preview_text, status)
    
    def _update_validation_display(self, errors, warnings):
        """Update the validation issues display."""
        self.validation_text.configure(state="normal")
        self.validation_text.delete("1.0", "end")
        
        if not errors and not warnings:
            self.validation_text.insert("1.0", "✅ No validation issues found. Data is ready for analysis.")
        else:
            content = []
            
            if errors:
                content.append("❌ ERRORS:")
                for error in errors:
                    content.append(f"  • {error}")
                content.append("")
            
            if warnings:
                content.append("⚠️ WARNINGS:")
                for warning in warnings:
                    content.append(f"  • {warning}")
            
            self.validation_text.insert("1.0", "\n".join(content))
        
        self.validation_text.configure(state="disabled")
    
    def _update_ready_status(self):
        """Update the ready status and continue button."""
        is_ready = self.data_controller.is_ready_for_analysis()
        
        if is_ready:
            self.ready_label.configure(
                text="✅ Data is configured and ready for topic modeling",
                text_color="green"
            )
            self.continue_button.configure(state="normal")
        else:
            if self.data_controller.current_data is None:
                status_text = "📁 Load a dataset file to begin"
            elif not self.data_controller.data_config.selected_columns:
                status_text = "📋 Select text columns for analysis"
            elif not self.data_controller.current_validation.is_valid:
                status_text = "⚠️ Fix validation errors before continuing"
            else:
                status_text = "⏳ Configuration incomplete"
            
            self.ready_label.configure(text=status_text)
            self.continue_button.configure(state="disabled")
    
    def _clear_data(self):
        """Clear all loaded data and reset the interface."""
        if messagebox.askyesno("Clear Data", "Are you sure you want to clear all loaded data?"):
            # Clear controller data
            self.data_controller.clear_data()
            
            # Reset UI components
            self.file_path_var.set("")
            self.file_info_label.configure(text="No file selected")
            self.load_button.configure(state="disabled")
            
            # Clear widgets
            self.data_preview_widget.update_data(None)
            self.column_selector_widget.set_columns([], {}, [])
            self.text_preview_widget.update_preview("", "No columns selected")
            self._update_validation_display([], [])
            self._update_ready_status()
            
            logger.info("Data cleared and UI reset")
    
    def _continue_to_next_step(self):
        """Continue to the next step (Model Configuration tab)."""
        if self.data_controller.is_ready_for_analysis():
            # Switch to Model Configuration tab
            try:
                # Get the parent tab view and switch tabs
                parent_window = self.parent
                while parent_window and not hasattr(parent_window, 'tab_view'):
                    parent_window = parent_window.master
                
                if parent_window and hasattr(parent_window, 'tab_view'):
                    parent_window.tab_view.set("Model Config")
                    logger.info("Switched to Model Configuration tab")
                else:
                    messagebox.showinfo("Next Step", "Data is ready! Please switch to the Model Configuration tab.")
                    
            except Exception as e:
                logger.error(f"Failed to switch tabs: {e}")
                messagebox.showinfo("Next Step", "Data is ready! Please switch to the Model Configuration tab.")
        else:
            messagebox.showwarning("Not Ready", "Please complete the data configuration before continuing.")
    
    def _on_status_update(self, status_message: str):
        """Handle status updates from the data controller."""
        # This could update a status bar or show temporary messages
        logger.info(f"Status update: {status_message}")
    
    def _on_progress_update(self, message: str, progress: float):
        """Handle progress updates from the data controller."""
        # This could update a progress bar
        logger.info(f"Progress update: {message} ({progress:.1%})")
    
    def get_data_controller(self) -> DataController:
        """Get the data controller instance for use by other components."""
        return self.data_controller
    
    def _bind_mousewheel(self, widget):
        """Bind mouse wheel events to the main scrollable frame."""
        def on_mousewheel(event):
            # Scroll the main frame
            try:
                if hasattr(widget, '_parent_canvas'):
                    widget._parent_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except:
                pass
        
        # Simple binding approach - the nested widgets will handle their own scrolling
        # and will prevent event bubbling when they're actively being scrolled
        try:
            widget.bind("<MouseWheel>", on_mousewheel, add=True)  # Windows/MacOS
            widget.bind("<Button-4>", lambda e: on_mousewheel(type('MockEvent', (), {'delta': 120})()), add=True)  # Linux scroll up
            widget.bind("<Button-5>", lambda e: on_mousewheel(type('MockEvent', (), {'delta': -120})()), add=True)  # Linux scroll down
        except:
            pass
    
    def _bind_validation_mousewheel(self, textbox):
        """Bind mouse wheel events specifically to the validation textbox."""
        def on_mousewheel(event):
            try:
                textbox.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except:
                pass
        
        try:
            textbox.bind("<MouseWheel>", on_mousewheel, add=True)  # Windows/MacOS
            textbox.bind("<Button-4>", lambda e: on_mousewheel(type('MockEvent', (), {'delta': 120})()), add=True)  # Linux scroll up
            textbox.bind("<Button-5>", lambda e: on_mousewheel(type('MockEvent', (), {'delta': -120})()), add=True)  # Linux scroll down
        except:
            pass 