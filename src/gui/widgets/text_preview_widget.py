"""
Text combination preview widget for showing combined text output.
"""
import customtkinter as ctk
from typing import Optional


class TextPreviewWidget(ctk.CTkFrame):
    """Widget for previewing how text columns will be combined."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.current_preview = ""
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the widget user interface."""
        # Header
        header_frame = ctk.CTkFrame(self)
        header_frame.pack(fill="x", padx=5, pady=(5, 0))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="Text Combination Preview",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title_label.pack(side="left", padx=10, pady=5)
        
        self.status_label = ctk.CTkLabel(
            header_frame,
            text="No columns selected",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.status_label.pack(side="right", padx=10, pady=5)
        
        # Preview area
        self.preview_text = ctk.CTkTextbox(
            self,
            height=150,
            wrap="word",
            state="disabled"
        )
        self.preview_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Bind mouse wheel for text preview
        self._bind_mousewheel(self.preview_text)
        
        # Instructions
        self.instructions_label = ctk.CTkLabel(
            self,
            text="Select columns above to see how text will be combined for analysis",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.instructions_label.pack(pady=(0, 5))
        
        # Initialize with empty state
        self.update_preview("", status="No columns selected")
    
    def update_preview(self, preview_text: str, status: Optional[str] = None):
        """
        Update the preview with new text.
        
        Args:
            preview_text: Combined text preview
            status: Status message to display
        """
        self.current_preview = preview_text
        
        # Update status
        if status:
            self.status_label.configure(text=status)
        
        # Update preview text
        self.preview_text.configure(state="normal")
        self.preview_text.delete("1.0", "end")
        
        if preview_text.strip():
            self.preview_text.insert("1.0", preview_text)
            self.instructions_label.configure(
                text="Preview of first few rows showing how selected columns will be combined"
            )
        else:
            placeholder_text = "No preview available.\n\nPlease:\n1. Load a data file\n2. Select text columns for analysis\n3. Configure combination settings"
            self.preview_text.insert("1.0", placeholder_text)
            self.instructions_label.configure(
                text="Select columns and configure settings to see preview"
            )
        
        self.preview_text.configure(state="disabled")
    
    def update_status(self, status: str):
        """Update just the status label."""
        self.status_label.configure(text=status)
    
    def _bind_mousewheel(self, widget):
        """Bind mouse wheel events to a widget for smooth scrolling."""
        def on_mousewheel(event):
            # Scroll the textbox - simpler approach
            try:
                widget.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except:
                pass
        
        # Bind to the textbox widget
        try:
            widget.bind("<MouseWheel>", on_mousewheel, add=True)  # Windows/MacOS
            widget.bind("<Button-4>", lambda e: on_mousewheel(type('MockEvent', (), {'delta': 120})()), add=True)  # Linux scroll up
            widget.bind("<Button-5>", lambda e: on_mousewheel(type('MockEvent', (), {'delta': -120})()), add=True)  # Linux scroll down
        except:
            pass  # Some widgets might not support binding
        
        # Also bind to the parent frame so scrolling works anywhere in the text preview area
        try:
            self.bind("<MouseWheel>", on_mousewheel, add=True)
            self.bind("<Button-4>", lambda e: on_mousewheel(type('MockEvent', (), {'delta': 120})()), add=True)
            self.bind("<Button-5>", lambda e: on_mousewheel(type('MockEvent', (), {'delta': -120})()), add=True)
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