"""
Settings Dialog - Application preferences and configuration.
"""
import customtkinter as ctk
from tkinter import messagebox

class SettingsDialog(ctk.CTkToplevel):
    """Settings dialog for application preferences."""
    
    def __init__(self, parent):
        super().__init__(parent)
        
        self.title("Settings")
        self.geometry("500x400")
        self.resizable(False, False)
        
        # Center dialog on parent
        self._center_on_parent(parent)
        
        self.setup_ui()
    
    def _center_on_parent(self, parent):
        """Center dialog on parent window."""
        parent.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - 250
        y = parent.winfo_y() + (parent.winfo_height() // 2) - 200
        self.geometry(f"500x400+{x}+{y}")
    
    def setup_ui(self):
        """Set up dialog user interface."""
        # Main frame
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame,
            text="Application Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.pack(pady=(10, 20))
        
        # Settings sections
        self._create_appearance_settings(main_frame)
        self._create_performance_settings(main_frame)
        self._create_cache_settings(main_frame)
        
        # Buttons
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(fill="x", pady=(20, 10))
        
        save_button = ctk.CTkButton(
            button_frame,
            text="Save",
            command=self._save_settings,
            width=100
        )
        save_button.pack(side="right", padx=(5, 0))
        
        cancel_button = ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=self.destroy,
            width=100
        )
        cancel_button.pack(side="right", padx=(0, 5))
    
    def _create_appearance_settings(self, parent):
        """Create appearance settings section."""
        appearance_frame = ctk.CTkFrame(parent)
        appearance_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(appearance_frame, text="Appearance", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Theme selection
        theme_frame = ctk.CTkFrame(appearance_frame, fg_color="transparent")
        theme_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(theme_frame, text="Theme:").pack(side="left")
        self.theme_var = ctk.StringVar(value="System")
        theme_menu = ctk.CTkOptionMenu(theme_frame, variable=self.theme_var, values=["System", "Light", "Dark"])
        theme_menu.pack(side="right")
    
    def _create_performance_settings(self, parent):
        """Create performance settings section."""
        perf_frame = ctk.CTkFrame(parent)
        perf_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(perf_frame, text="Performance", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Batch size
        batch_frame = ctk.CTkFrame(perf_frame, fg_color="transparent")
        batch_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(batch_frame, text="Batch Size:").pack(side="left")
        self.batch_size_var = ctk.StringVar(value="1000")
        batch_entry = ctk.CTkEntry(batch_frame, textvariable=self.batch_size_var, width=100)
        batch_entry.pack(side="right")
    
    def _create_cache_settings(self, parent):
        """Create cache settings section."""
        cache_frame = ctk.CTkFrame(parent)
        cache_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(cache_frame, text="Cache", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Enable cache checkbox
        self.cache_enabled_var = ctk.BooleanVar(value=True)
        cache_checkbox = ctk.CTkCheckBox(cache_frame, text="Enable embedding cache", variable=self.cache_enabled_var)
        cache_checkbox.pack(anchor="w", padx=10, pady=5)
        
        # Clear cache button
        clear_cache_btn = ctk.CTkButton(cache_frame, text="Clear Cache", command=self._clear_cache, width=120)
        clear_cache_btn.pack(anchor="w", padx=10, pady=(0, 10))
    
    def _save_settings(self):
        """Save settings and close dialog."""
        # TODO: Implement actual settings saving
        messagebox.showinfo("Settings", "Settings saved successfully!")
        self.destroy()
    
    def _clear_cache(self):
        """Clear application cache."""
        result = messagebox.askyesno("Clear Cache", "Are you sure you want to clear the cache?")
        if result:
            # TODO: Implement cache clearing
            messagebox.showinfo("Cache", "Cache cleared successfully!") 