"""
Column selector widget for choosing and configuring text columns.
"""
import customtkinter as ctk
from typing import List, Dict, Any, Callable, Optional


class ColumnSelectorWidget(ctk.CTkFrame):
    """Widget for selecting and configuring text columns for analysis."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.available_columns: List[str] = []
        self.selected_columns: List[str] = []
        self.column_info: Dict[str, Any] = {}
        self.recommended_columns: List[str] = []
        
        # Callbacks
        self.on_selection_changed: Optional[Callable[[List[str]], None]] = None
        self.on_settings_changed: Optional[Callable[[], None]] = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the widget user interface."""
        # Header
        header_frame = ctk.CTkFrame(self)
        header_frame.pack(fill="x", padx=5, pady=(5, 0))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="Column Selection",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title_label.pack(side="left", padx=10, pady=5)
        
        # Auto-select recommended button
        self.auto_select_btn = ctk.CTkButton(
            header_frame,
            text="Auto-select Recommended",
            width=160,
            command=self._auto_select_recommended
        )
        self.auto_select_btn.pack(side="right", padx=10, pady=5)
        
        # Main content frame
        content_frame = ctk.CTkFrame(self)
        content_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Left side - Available columns
        left_frame = ctk.CTkFrame(content_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(5, 2), pady=5)
        
        ctk.CTkLabel(
            left_frame,
            text="Available Columns",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(pady=(5, 2))
        
        # Available columns listbox
        self.available_listbox = ctk.CTkScrollableFrame(left_frame, height=200)
        self.available_listbox.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Bind mouse wheel for available columns
        self._bind_mousewheel(self.available_listbox)
        
        # Middle - Controls
        middle_frame = ctk.CTkFrame(content_frame)
        middle_frame.pack(side="left", fill="y", padx=2, pady=5)
        
        self.add_btn = ctk.CTkButton(
            middle_frame,
            text="→",
            width=40,
            command=self._add_selected_column
        )
        self.add_btn.pack(pady=5)
        
        self.remove_btn = ctk.CTkButton(
            middle_frame,
            text="←",
            width=40,
            command=self._remove_selected_column
        )
        self.remove_btn.pack(pady=5)
        
        self.clear_btn = ctk.CTkButton(
            middle_frame,
            text="Clear",
            width=60,
            command=self._clear_selection
        )
        self.clear_btn.pack(pady=20)
        
        # Right side - Selected columns
        right_frame = ctk.CTkFrame(content_frame)
        right_frame.pack(side="left", fill="both", expand=True, padx=(2, 5), pady=5)
        
        ctk.CTkLabel(
            right_frame,
            text="Selected Columns",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(pady=(5, 2))
        
        # Selected columns listbox
        self.selected_listbox = ctk.CTkScrollableFrame(right_frame, height=200)
        self.selected_listbox.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Bind mouse wheel for selected columns
        self._bind_mousewheel(self.selected_listbox)
        
        # Configuration frame
        config_frame = ctk.CTkFrame(self)
        config_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(
            config_frame,
            text="Text Combination Settings",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=10, pady=(5, 2))
        
        # Combination settings
        settings_frame = ctk.CTkFrame(config_frame)
        settings_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Separator
        ctk.CTkLabel(settings_frame, text="Separator:").pack(side="left", padx=(5, 2), pady=5)
        
        self.separator_var = ctk.StringVar(value=" ")
        self.separator_entry = ctk.CTkEntry(
            settings_frame,
            textvariable=self.separator_var,
            width=50
        )
        self.separator_entry.pack(side="left", padx=2, pady=5)
        self.separator_var.trace("w", self._on_settings_changed)
        
        # Include column names checkbox
        self.include_names_var = ctk.BooleanVar(value=False)
        self.include_names_cb = ctk.CTkCheckBox(
            settings_frame,
            text="Include column names",
            variable=self.include_names_var,
            command=self._on_settings_changed
        )
        self.include_names_cb.pack(side="left", padx=20, pady=5)
        
        # Initialize empty state
        self._update_available_display()
        self._update_selected_display()
    
    def set_columns(self, columns: List[str], column_info: Dict[str, Any], recommended: List[str]):
        """
        Set available columns and their information.
        
        Args:
            columns: List of available column names
            column_info: Dictionary with column analysis information
            recommended: List of recommended column names
        """
        self.available_columns = columns
        self.column_info = column_info
        self.recommended_columns = recommended
        self.selected_columns = []
        
        self._update_available_display()
        self._update_selected_display()
        
        # Enable auto-select button if there are recommendations
        if recommended:
            self.auto_select_btn.configure(state="normal")
        else:
            self.auto_select_btn.configure(state="disabled")
    
    def get_selected_columns(self) -> List[str]:
        """Get currently selected columns."""
        return self.selected_columns.copy()
    
    def get_combination_settings(self) -> Dict[str, Any]:
        """Get text combination settings."""
        return {
            'separator': self.separator_var.get(),
            'include_column_names': self.include_names_var.get()
        }
    
    def set_selection_callback(self, callback: Callable[[List[str]], None]):
        """Set callback for selection changes."""
        self.on_selection_changed = callback
    
    def set_settings_callback(self, callback: Callable[[], None]):
        """Set callback for settings changes."""
        self.on_settings_changed = callback
    
    def _update_available_display(self):
        """Update the available columns display."""
        # Clear existing widgets
        for widget in self.available_listbox.winfo_children():
            widget.destroy()
        
        if not self.available_columns:
            empty_label = ctk.CTkLabel(
                self.available_listbox,
                text="No columns available",
                text_color="gray"
            )
            empty_label.pack(pady=20)
            return
        
        # Sort columns: recommended first, then alphabetical
        sorted_columns = []
        for col in self.recommended_columns:
            if col in self.available_columns:
                sorted_columns.append(col)
        
        for col in sorted(self.available_columns):
            if col not in sorted_columns:
                sorted_columns.append(col)
        
        # Create column widgets
        for col in sorted_columns:
            self._create_column_widget(self.available_listbox, col, is_available=True)
    
    def _update_selected_display(self):
        """Update the selected columns display."""
        # Clear existing widgets
        for widget in self.selected_listbox.winfo_children():
            widget.destroy()
        
        if not self.selected_columns:
            empty_label = ctk.CTkLabel(
                self.selected_listbox,
                text="No columns selected",
                text_color="gray"
            )
            empty_label.pack(pady=20)
            return
        
        # Create selected column widgets
        for col in self.selected_columns:
            self._create_column_widget(self.selected_listbox, col, is_available=False)
    
    def _create_column_widget(self, parent, column_name: str, is_available: bool):
        """Create a widget for a single column."""
        # Determine styling based on recommendation
        is_recommended = column_name in self.recommended_columns
        is_selected_elsewhere = column_name in self.selected_columns and is_available
        
        if is_recommended:
            fg_color = "#1f538d"  # Blue for recommended
        elif is_selected_elsewhere:
            fg_color = "gray"  # Gray for already selected
        else:
            fg_color = None  # Default
        
        # Create frame for column
        col_frame = ctk.CTkFrame(parent, fg_color=fg_color)
        col_frame.pack(fill="x", padx=2, pady=1)
        
        # Column name and info
        name_label = ctk.CTkLabel(
            col_frame,
            text=column_name,
            font=ctk.CTkFont(weight="bold" if is_recommended else "normal"),
            anchor="w"
        )
        name_label.pack(side="left", padx=(10, 5), pady=5, fill="x", expand=True)
        
        # Column info
        if column_name in self.column_info:
            col_info = self.column_info[column_name]
            info_parts = []
            
            if 'data_type' in col_info:
                info_parts.append(col_info['data_type'])
            
            if 'missing_percentage' in col_info and col_info['missing_percentage'] > 0:
                info_parts.append(f"{col_info['missing_percentage']:.0f}% missing")
            
            if 'suggested_for_analysis' in col_info and not col_info['suggested_for_analysis']:
                info_parts.append("not recommended")
            
            if info_parts:
                info_text = " | ".join(info_parts)
                info_label = ctk.CTkLabel(
                    col_frame,
                    text=info_text,
                    font=ctk.CTkFont(size=10),
                    text_color="lightgray"
                )
                info_label.pack(side="right", padx=(5, 10), pady=5)
        
        # Make clickable if available
        if is_available and not is_selected_elsewhere:
            col_frame.bind("<Button-1>", lambda e, col=column_name: self._toggle_column_selection(col))
            name_label.bind("<Button-1>", lambda e, col=column_name: self._toggle_column_selection(col))
    
    def _toggle_column_selection(self, column_name: str):
        """Toggle column selection."""
        if column_name in self.selected_columns:
            self.selected_columns.remove(column_name)
        else:
            self.selected_columns.append(column_name)
        
        self._update_available_display()
        self._update_selected_display()
        self._notify_selection_changed()
    
    def _add_selected_column(self):
        """Add first available column (for button interface)."""
        available = [col for col in self.available_columns if col not in self.selected_columns]
        if available:
            self.selected_columns.append(available[0])
            self._update_available_display()
            self._update_selected_display()
            self._notify_selection_changed()
    
    def _remove_selected_column(self):
        """Remove last selected column (for button interface)."""
        if self.selected_columns:
            self.selected_columns.pop()
            self._update_available_display()
            self._update_selected_display()
            self._notify_selection_changed()
    
    def _clear_selection(self):
        """Clear all selected columns."""
        self.selected_columns = []
        self._update_available_display()
        self._update_selected_display()
        self._notify_selection_changed()
    
    def _auto_select_recommended(self):
        """Auto-select all recommended columns."""
        self.selected_columns = [col for col in self.recommended_columns if col in self.available_columns]
        self._update_available_display()
        self._update_selected_display()
        self._notify_selection_changed()
    
    def _notify_selection_changed(self):
        """Notify about selection changes."""
        if self.on_selection_changed:
            self.on_selection_changed(self.selected_columns)
    
    def _on_settings_changed(self, *args):
        """Handle settings changes."""
        if self.on_settings_changed:
            self.on_settings_changed()
    
    def _bind_mousewheel(self, widget):
        """Bind mouse wheel events to a widget for smooth scrolling."""
        def on_mousewheel(event):
            # Scroll the widget - simpler approach without complex detection
            try:
                if hasattr(widget, '_parent_canvas'):
                    widget._parent_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except:
                pass
        
        # Bind to the widget and all its children recursively
        def bind_to_widget_and_children(w):
            try:
                w.bind("<MouseWheel>", on_mousewheel, add=True)  # Windows/MacOS
                w.bind("<Button-4>", lambda e: on_mousewheel(type('MockEvent', (), {'delta': 120})()), add=True)  # Linux scroll up
                w.bind("<Button-5>", lambda e: on_mousewheel(type('MockEvent', (), {'delta': -120})()), add=True)  # Linux scroll down
            except:
                pass  # Some widgets might not support binding
            
            # Recursively bind to all children
            try:
                for child in w.winfo_children():
                    bind_to_widget_and_children(child)
            except:
                pass
        
        # Bind to the main widget
        bind_to_widget_and_children(widget)
        
        # Also bind to the containing frame
        try:
            # Find the parent frame that contains this scrollable widget
            parent_frame = widget.master
            if parent_frame:
                parent_frame.bind("<MouseWheel>", on_mousewheel, add=True)
                parent_frame.bind("<Button-4>", lambda e: on_mousewheel(type('MockEvent', (), {'delta': 120})()), add=True)
                parent_frame.bind("<Button-5>", lambda e: on_mousewheel(type('MockEvent', (), {'delta': -120})()), add=True)
        except:
            pass
    
    def _is_child_of(self, widget, parent):
        """Check if widget is a child of parent."""
        try:
            current = widget
            while current:
                if current == parent:
                    return True
                current = current.master
            return False
        except:
            return False 