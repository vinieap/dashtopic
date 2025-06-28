"""
BERTopic Desktop Application
Main entry point for the application.
"""
import sys
import logging
import customtkinter as ctk
from src.gui.main_window import MainWindow
from src.utils.logging_utils import setup_logging

def main():
    """Main application entry point."""
    setup_logging()
    
    # Set CustomTkinter appearance mode and color theme
    ctk.set_appearance_mode("System")  # Modes: "System", "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"
    
    # Create and run the application
    app = MainWindow()
    app.mainloop()

if __name__ == "__main__":
    main() 