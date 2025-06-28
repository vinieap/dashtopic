"""
About Dialog - Application information and credits.
"""
import customtkinter as ctk
from ...utils.constants import APP_NAME, APP_VERSION, APP_DESCRIPTION

class AboutDialog(ctk.CTkToplevel):
    """About dialog showing application information."""
    
    def __init__(self, parent):
        super().__init__(parent)
        
        self.title("About")
        self.geometry("400x300")
        self.resizable(False, False)
        
        # Center dialog on parent
        self._center_on_parent(parent)
        
        self.setup_ui()
    
    def _center_on_parent(self, parent):
        """Center dialog on parent window."""
        parent.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - 200
        y = parent.winfo_y() + (parent.winfo_height() // 2) - 150
        self.geometry(f"400x300+{x}+{y}")
    
    def setup_ui(self):
        """Set up dialog user interface."""
        # Main frame
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # App name
        app_name_label = ctk.CTkLabel(
            main_frame,
            text=APP_NAME,
            font=ctk.CTkFont(size=18, weight="bold")
        )
        app_name_label.pack(pady=(10, 5))
        
        # Version
        version_label = ctk.CTkLabel(
            main_frame,
            text=f"Version {APP_VERSION}",
            font=ctk.CTkFont(size=12)
        )
        version_label.pack(pady=5)
        
        # Description
        desc_label = ctk.CTkLabel(
            main_frame,
            text=APP_DESCRIPTION,
            font=ctk.CTkFont(size=11),
            wraplength=350
        )
        desc_label.pack(pady=15)
        
        # Credits
        credits_label = ctk.CTkLabel(
            main_frame,
            text="Built with:\n• BERTopic\n• SentenceTransformers\n• CustomTkinter\n• Python",
            font=ctk.CTkFont(size=10),
            justify="left"
        )
        credits_label.pack(pady=10)
        
        # Close button
        close_button = ctk.CTkButton(
            main_frame,
            text="Close",
            command=self.destroy,
            width=100
        )
        close_button.pack(pady=(20, 10)) 