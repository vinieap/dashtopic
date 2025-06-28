"""
Data preview widget for displaying tabular data with statistics.
"""
import customtkinter as ctk
import pandas as pd
from tkinter import ttk
from typing import Optional, Dict, Any


class DataPreviewWidget(ctk.CTkFrame):
    """Widget for previewing data with statistics and basic filtering."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.current_data: Optional[pd.DataFrame] = None
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the widget user interface."""
        # Header with search
        header_frame = ctk.CTkFrame(self)
        header_frame.pack(fill="x", padx=5, pady=(5, 2))
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame,
            text="Data Preview",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title_label.pack(side="left", padx=10, pady=5)
        
        # Search box
        self.search_var = ctk.StringVar()
        self.search_var.trace("w", self._on_search_changed)
        
        search_entry = ctk.CTkEntry(
            header_frame,
            textvariable=self.search_var,
            placeholder_text="Search data...",
            width=200
        )
        search_entry.pack(side="right", padx=10, pady=5)
        
        # Info label
        self.info_label = ctk.CTkLabel(
            self,
            text="No data loaded",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.info_label.pack(fill="x", padx=5, pady=2)
        
        # Statistics frame
        self.stats_frame = ctk.CTkFrame(self)
        self.stats_frame.pack(fill="x", padx=5, pady=2)
        
        # Create scrollable frame for data table
        self.scrollable_frame = ctk.CTkScrollableFrame(self, height=300)
        self.scrollable_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Bind mouse wheel events for smooth scrolling
        self._bind_mousewheel(self.scrollable_frame)
        
        # Initially show empty state
        self.empty_label = ctk.CTkLabel(
            self.scrollable_frame,
            text="No data to preview",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.empty_label.pack(expand=True, pady=50)
    
    def _bind_mousewheel(self, widget):
        """Bind mouse wheel events to a widget for smooth scrolling."""
        def on_mousewheel(event):
            # Scroll the widget regardless of exact position - if we're in this area, scroll it
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
        
        # Also bind to the parent scrollable frame itself
        try:
            self.bind("<MouseWheel>", on_mousewheel, add=True)  
            self.bind("<Button-4>", lambda e: on_mousewheel(type('MockEvent', (), {'delta': 120})()), add=True)
            self.bind("<Button-5>", lambda e: on_mousewheel(type('MockEvent', (), {'delta': -120})()), add=True)
        except:
            pass
    
    def update_data(self, df: Optional[pd.DataFrame], summary: Optional[Dict[str, Any]] = None):
        """
        Update the widget with new data.
        
        Args:
            df: DataFrame to preview
            summary: Optional summary statistics
        """
        self.current_data = df
        
        # Clear existing content
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        for widget in self.stats_frame.winfo_children():
            widget.destroy()
        
        if df is None or df.empty:
            self.info_label.configure(text="No data loaded")
            self.empty_label = ctk.CTkLabel(
                self.scrollable_frame,
                text="No data to preview",
                font=ctk.CTkFont(size=14),
                text_color="gray"
            )
            self.empty_label.pack(expand=True, pady=50)
            return
        
        # Update info label
        self.info_label.configure(text=f"{len(df):,} rows × {len(df.columns)} columns")
        
        # Show statistics
        self._create_statistics_display(summary)
        
        # Show data table
        self._create_data_table(df)
    
    def _create_statistics_display(self, summary: Optional[Dict[str, Any]]):
        """Create statistics display."""
        if not summary:
            return
        
        # File info
        if 'file_info' in summary:
            file_info = summary['file_info']
            file_text = f"File: {file_info.get('name', 'Unknown')} ({file_info.get('format', 'Unknown')}, {file_info.get('size_mb', 0):.1f} MB)"
            
            file_label = ctk.CTkLabel(
                self.stats_frame,
                text=file_text,
                font=ctk.CTkFont(size=11)
            )
            file_label.pack(side="left", padx=10, pady=5)
        
        # Data info
        if 'data_info' in summary:
            data_info = summary['data_info']
            data_text = f"Memory: {data_info.get('memory_usage_mb', 0):.1f} MB | Text columns: {data_info.get('text_columns', 0)}"
            
            data_label = ctk.CTkLabel(
                self.stats_frame,
                text=data_text,
                font=ctk.CTkFont(size=11)
            )
            data_label.pack(side="right", padx=10, pady=5)
    
    def _create_data_table(self, df: pd.DataFrame):
        """Create data table display."""
        # Limit displayed rows for performance
        display_df = df.head(50)
        
        # Create table frame
        table_frame = ctk.CTkFrame(self.scrollable_frame)
        table_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Headers
        header_frame = ctk.CTkFrame(table_frame)
        header_frame.pack(fill="x", padx=2, pady=2)
        
        # Row number header
        ctk.CTkLabel(
            header_frame,
            text="#",
            font=ctk.CTkFont(weight="bold"),
            width=40
        ).pack(side="left", padx=1)
        
        # Column headers
        for col in display_df.columns:
            header_label = ctk.CTkLabel(
                header_frame,
                text=str(col)[:20] + ("..." if len(str(col)) > 20 else ""),
                font=ctk.CTkFont(weight="bold"),
                width=120
            )
            header_label.pack(side="left", padx=1)
        
        # Data rows
        for idx, (_, row) in enumerate(display_df.iterrows()):
            if idx >= 50:  # Limit display for performance
                break
                
            row_frame = ctk.CTkFrame(table_frame)
            row_frame.pack(fill="x", padx=2, pady=1)
            
            # Row number
            ctk.CTkLabel(
                row_frame,
                text=str(idx),
                width=40,
                font=ctk.CTkFont(size=10)
            ).pack(side="left", padx=1)
            
            # Cell values
            for col in display_df.columns:
                value = row[col]
                if pd.isna(value):
                    display_value = "NaN"
                    text_color = "gray"
                else:
                    display_value = str(value)[:30] + ("..." if len(str(value)) > 30 else "")
                    text_color = None
                
                cell_label = ctk.CTkLabel(
                    row_frame,
                    text=display_value,
                    width=120,
                    font=ctk.CTkFont(size=10),
                    text_color=text_color
                )
                cell_label.pack(side="left", padx=1)
        
        # Show truncation notice if needed
        if len(df) > 50:
            notice_label = ctk.CTkLabel(
                table_frame,
                text=f"Showing first 50 rows of {len(df):,} total rows",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            )
            notice_label.pack(pady=5)
    
    def _on_search_changed(self, *args):
        """Handle search text changes."""
        if self.current_data is None:
            return
        
        search_text = self.search_var.get().lower().strip()
        
        if not search_text:
            # Show original data
            self._create_data_table(self.current_data)
            return
        
        try:
            # Filter data based on search
            mask = self.current_data.astype(str).apply(
                lambda x: x.str.lower().str.contains(search_text, na=False)
            ).any(axis=1)
            
            filtered_df = self.current_data[mask]
            
            # Update table with filtered data
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
            
            if filtered_df.empty:
                no_results_label = ctk.CTkLabel(
                    self.scrollable_frame,
                    text=f"No results found for '{search_text}'",
                    font=ctk.CTkFont(size=12),
                    text_color="gray"
                )
                no_results_label.pack(expand=True, pady=50)
            else:
                self._create_data_table(filtered_df)
                
                # Show filter info
                filter_info = ctk.CTkLabel(
                    self.scrollable_frame,
                    text=f"Showing {len(filtered_df):,} of {len(self.current_data):,} rows matching '{search_text}'",
                    font=ctk.CTkFont(size=10),
                    text_color="blue"
                )
                filter_info.pack(pady=5)
        
        except Exception as e:
            # Handle search errors gracefully
            error_label = ctk.CTkLabel(
                self.scrollable_frame,
                text=f"Search error: {str(e)}",
                font=ctk.CTkFont(size=12),
                text_color="red"
            )
            error_label.pack(expand=True, pady=50) 