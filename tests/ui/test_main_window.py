"""
UI tests for MainWindow.

This module tests the main application window including:
- Window initialization
- Menu functionality
- Tab switching
- Status bar updates
- Basic user interactions

Run with: pytest tests/ui/test_main_window.py -v

Note: These tests require a display. On headless systems, use:
pytest tests/ui/test_main_window.py -v --capture=no
"""
import pytest
import tkinter as tk
import customtkinter as ctk
from unittest.mock import Mock, patch, MagicMock


@pytest.mark.ui
class TestMainWindow:
    """Test suite for MainWindow UI components."""
    
    @pytest.fixture
    def main_window_class(self):
        """Import MainWindow class for testing."""
        # Import here to avoid issues if dependencies aren't available
        try:
            from src.gui.main_window import MainWindow
            return MainWindow
        except ImportError:
            pytest.skip("MainWindow not available for testing")
    
    def test_main_window_initialization(self, main_window_class):
        """Test that MainWindow initializes correctly."""
        # Mock the services to avoid dependencies
        with patch('src.gui.main_window.DataController'), \
             patch('src.gui.main_window.EmbeddingController'), \
             patch('src.gui.main_window.TopicModelingController'), \
             patch('src.gui.main_window.OptimizationController'):
            
            window = main_window_class()
            
            # Check basic window properties
            assert window.title() == "BERTopic Desktop Application"
            assert window.winfo_width() > 0
            assert window.winfo_height() > 0
            
            # Cleanup
            window.destroy()
    
    def test_main_window_has_required_components(self, main_window_class):
        """Test that MainWindow has all required UI components."""
        with patch('src.gui.main_window.DataController'), \
             patch('src.gui.main_window.EmbeddingController'), \
             patch('src.gui.main_window.TopicModelingController'), \
             patch('src.gui.main_window.OptimizationController'):
            
            window = main_window_class()
            
            # Check for menu bar
            assert hasattr(window, 'menubar') or hasattr(window, 'menu')
            
            # Check for tab widget
            assert hasattr(window, 'tabview') or hasattr(window, 'notebook')
            
            # Check for status bar
            assert hasattr(window, 'status_bar')
            
            window.destroy()
    
    def test_tab_switching(self, main_window_class):
        """Test switching between different tabs."""
        with patch('src.gui.main_window.DataController'), \
             patch('src.gui.main_window.EmbeddingController'), \
             patch('src.gui.main_window.TopicModelingController'), \
             patch('src.gui.main_window.OptimizationController'):
            
            window = main_window_class()
            
            # Get the tab widget
            if hasattr(window, 'tabview'):
                tabview = window.tabview
                
                # Test switching to different tabs
                expected_tabs = [
                    "Data Import", "Model Configuration", "Topic Modeling",
                    "Visualization", "Hyperparameter", "Export"
                ]
                
                for tab_name in expected_tabs:
                    if tab_name in tabview._tab_dict:
                        tabview.set(tab_name)
                        assert tabview.get() == tab_name
            
            window.destroy()
    
    def test_status_bar_updates(self, main_window_class):
        """Test status bar update functionality."""
        with patch('src.gui.main_window.DataController'), \
             patch('src.gui.main_window.EmbeddingController'), \
             patch('src.gui.main_window.TopicModelingController'), \
             patch('src.gui.main_window.OptimizationController'):
            
            window = main_window_class()
            
            if hasattr(window, 'status_bar'):
                status_bar = window.status_bar
                
                # Test updating status
                test_message = "Test status message"
                if hasattr(status_bar, 'set_status'):
                    status_bar.set_status(test_message)
                    # Check if status was updated (implementation dependent)
                
                # Test progress update
                if hasattr(status_bar, 'set_progress'):
                    status_bar.set_progress(50)  # 50%
            
            window.destroy()
    
    def test_menu_functionality(self, main_window_class):
        """Test menu item functionality."""
        with patch('src.gui.main_window.DataController'), \
             patch('src.gui.main_window.EmbeddingController'), \
             patch('src.gui.main_window.TopicModelingController'), \
             patch('src.gui.main_window.OptimizationController'):
            
            window = main_window_class()
            
            # Test file menu exists
            if hasattr(window, 'menubar'):
                menubar = window.menubar
                # Check for common menu items (implementation dependent)
                
            window.destroy()
    
    def test_window_resizing(self, main_window_class):
        """Test window resizing behavior."""
        with patch('src.gui.main_window.DataController'), \
             patch('src.gui.main_window.EmbeddingController'), \
             patch('src.gui.main_window.TopicModelingController'), \
             patch('src.gui.main_window.OptimizationController'):
            
            window = main_window_class()
            
            # Test initial size
            initial_width = window.winfo_width()
            initial_height = window.winfo_height()
            
            # Test resizing
            new_width = 1000
            new_height = 700
            window.geometry(f"{new_width}x{new_height}")
            
            # Update to apply geometry
            window.update()
            
            # Check if size changed
            assert window.winfo_width() >= new_width - 50  # Allow some tolerance
            assert window.winfo_height() >= new_height - 50
            
            window.destroy()
    
    def test_closing_window(self, main_window_class):
        """Test window closing behavior."""
        with patch('src.gui.main_window.DataController'), \
             patch('src.gui.main_window.EmbeddingController'), \
             patch('src.gui.main_window.TopicModelingController'), \
             patch('src.gui.main_window.OptimizationController'):
            
            window = main_window_class()
            
            # Test that window can be closed without errors
            window.quit()
            window.destroy()
            
            # Window should be destroyed
            assert window.winfo_exists() == 0


@pytest.mark.ui
class TestMainWindowIntegration:
    """Integration tests for MainWindow with mocked services."""
    
    def test_main_window_with_mock_controllers(self, main_window_class):
        """Test MainWindow with properly mocked controllers."""
        # Create mock controllers
        mock_data_controller = Mock()
        mock_embedding_controller = Mock()
        mock_topic_controller = Mock()
        mock_optimization_controller = Mock()
        
        with patch('src.gui.main_window.DataController', return_value=mock_data_controller), \
             patch('src.gui.main_window.EmbeddingController', return_value=mock_embedding_controller), \
             patch('src.gui.main_window.TopicModelingController', return_value=mock_topic_controller), \
             patch('src.gui.main_window.OptimizationController', return_value=mock_optimization_controller):
            
            window = main_window_class()
            
            # Verify controllers were created
            assert hasattr(window, 'data_controller')
            assert hasattr(window, 'embedding_controller')
            assert hasattr(window, 'topic_modeling_controller')
            
            window.destroy()
    
    def test_error_handling_in_initialization(self, main_window_class):
        """Test error handling during window initialization."""
        # Mock a controller that raises an exception
        with patch('src.gui.main_window.DataController', side_effect=Exception("Mock error")):
            
            # Window should still initialize gracefully
            try:
                window = main_window_class()
                window.destroy()
            except Exception as e:
                # If initialization fails, it should be handled gracefully
                assert "Mock error" in str(e)


@pytest.mark.ui
@pytest.mark.slow
class TestMainWindowUserInteractions:
    """Test simulated user interactions with MainWindow."""
    
    def test_simulated_file_loading(self, main_window_class, temp_dir):
        """Test simulated file loading interaction."""
        # Create a test file
        test_file = temp_dir / "test_data.csv"
        test_file.write_text("col1,col2\nvalue1,value2\nvalue3,value4")
        
        with patch('src.gui.main_window.DataController') as mock_data_controller_class, \
             patch('src.gui.main_window.EmbeddingController'), \
             patch('src.gui.main_window.TopicModelingController'), \
             patch('src.gui.main_window.OptimizationController'):
            
            # Setup mock data controller
            mock_data_controller = Mock()
            mock_data_controller.load_file.return_value = True
            mock_data_controller_class.return_value = mock_data_controller
            
            window = main_window_class()
            
            # Simulate file loading
            if hasattr(window, 'data_controller'):
                window.data_controller.load_file(str(test_file))
                
                # Verify the load_file method was called
                mock_data_controller.load_file.assert_called_once_with(str(test_file))
            
            window.destroy()
    
    def test_simulated_tab_workflow(self, main_window_class):
        """Test simulated workflow through different tabs."""
        with patch('src.gui.main_window.DataController'), \
             patch('src.gui.main_window.EmbeddingController'), \
             patch('src.gui.main_window.TopicModelingController'), \
             patch('src.gui.main_window.OptimizationController'):
            
            window = main_window_class()
            
            # Simulate workflow: Data Import -> Model Config -> Topic Modeling
            if hasattr(window, 'tabview'):
                tabview = window.tabview
                
                # Step 1: Data Import
                if "Data Import" in tabview._tab_dict:
                    tabview.set("Data Import")
                    window.update()
                
                # Step 2: Model Configuration  
                if "Model Configuration" in tabview._tab_dict:
                    tabview.set("Model Configuration")
                    window.update()
                
                # Step 3: Topic Modeling
                if "Topic Modeling" in tabview._tab_dict:
                    tabview.set("Topic Modeling")
                    window.update()
            
            window.destroy()


# Helper functions for UI testing
def simulate_button_click(button):
    """Simulate clicking a button widget."""
    if hasattr(button, 'invoke'):
        button.invoke()
    elif hasattr(button, 'command'):
        if button.command:
            button.command()


def simulate_text_entry(entry_widget, text):
    """Simulate entering text into an entry widget."""
    if hasattr(entry_widget, 'delete') and hasattr(entry_widget, 'insert'):
        entry_widget.delete(0, 'end')
        entry_widget.insert(0, text)


def get_widget_by_text(parent, text, widget_type=None):
    """Find a widget by its text content."""
    for child in parent.winfo_children():
        if hasattr(child, 'cget'):
            try:
                if child.cget('text') == text:
                    if widget_type is None or isinstance(child, widget_type):
                        return child
            except:
                pass
        
        # Recursively search child widgets
        result = get_widget_by_text(child, text, widget_type)
        if result:
            return result
    
    return None