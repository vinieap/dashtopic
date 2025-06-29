"""
Data controller for managing data import and validation workflows.
"""
import logging
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any, Callable

from ..models.data_models import DataConfig, FileMetadata, ValidationResult
from ..services.file_io_service import FileIOService
from ..services.data_validation_service import DataValidationService

logger = logging.getLogger(__name__)


class DataController:
    """Controller for managing data import and processing workflows."""
    
    def __init__(self):
        self.file_io_service = FileIOService()
        self.validation_service = DataValidationService()
        self.current_data: Optional[pd.DataFrame] = None
        self.current_metadata: Optional[FileMetadata] = None
        self.current_validation: Optional[ValidationResult] = None
        self.data_config = DataConfig()
        
        # Callbacks for UI updates
        self.on_progress_update: Optional[Callable[[str, float], None]] = None
        self.on_status_update: Optional[Callable[[str], None]] = None
    
    def set_progress_callback(self, callback: Callable[[str, float], None]):
        """Set callback for progress updates."""
        self.on_progress_update = callback
    
    def set_status_callback(self, callback: Callable[[str], None]):
        """Set callback for status updates."""
        self.on_status_update = callback
    
    def _update_progress(self, message: str, progress: float):
        """Update progress if callback is set."""
        if self.on_progress_update:
            self.on_progress_update(message, progress)
    
    def _update_status(self, message: str):
        """Update status if callback is set."""
        if self.on_status_update:
            self.on_status_update(message)
        logger.info(message)
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get basic file information without loading the full file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        try:
            return self.file_io_service.get_file_info(file_path)
        except Exception as e:
            logger.error(f"Failed to get file info: {e}")
            raise
    
    def load_file(self, file_path: str, **kwargs) -> Tuple[bool, str]:
        """
        Load file and perform validation.
        
        Args:
            file_path: Path to the file to load
            **kwargs: Additional arguments for file loading
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            self._update_status("Loading file...")
            self._update_progress("Reading file...", 0.1)
            
            # Load the file
            df, metadata = self.file_io_service.load_file(file_path, **kwargs)
            
            self._update_progress("Validating data...", 0.5)
            
            # Validate the loaded data
            validation_result = self.validation_service.validate_file_data(df, metadata)
            
            self._update_progress("Processing complete", 1.0)
            
            # Store the results
            self.current_data = df
            self.current_metadata = metadata
            self.current_validation = validation_result
            
            # Update data config
            self.data_config.file_metadata = metadata
            
            # Generate status message
            status_msg = f"Loaded {metadata.file_name}: {metadata.row_count:,} rows, {metadata.column_count} columns"
            
            if not validation_result.is_valid:
                status_msg += f" (with {len(validation_result.errors)} errors)"
            elif validation_result.warnings:
                status_msg += f" (with {len(validation_result.warnings)} warnings)"
            
            self._update_status(status_msg)
            return True, status_msg
            
        except Exception as e:
            error_msg = f"Failed to load file: {str(e)}"
            self._update_status(error_msg)
            logger.error(f"File loading error: {e}")
            return False, error_msg
    
    def get_data_preview(self, max_rows: int = 10) -> Optional[pd.DataFrame]:
        """
        Get preview of loaded data.
        
        Args:
            max_rows: Maximum number of rows to return
            
        Returns:
            DataFrame with preview data or None if no data loaded
        """
        if self.current_data is None:
            return None
        
        return self.current_data.head(max_rows)
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for loaded data.
        
        Returns:
            Dictionary with data summary information
        """
        if self.current_data is None or self.current_metadata is None:
            return {}
        
        summary = {
            'file_info': {
                'name': self.current_metadata.file_name,
                'format': self.current_metadata.file_format,
                'size_mb': self.current_metadata.file_size_mb,
                'encoding': self.current_metadata.encoding
            },
            'data_info': {
                'rows': self.current_metadata.row_count,
                'columns': self.current_metadata.column_count,
                'text_columns': len(self.current_metadata.text_columns),
                'memory_usage_mb': self.current_data.memory_usage(deep=True).sum() / (1024 * 1024)
            }
        }
        
        # Add validation info if available
        if self.current_validation:
            summary['validation'] = {
                'is_valid': self.current_validation.is_valid,
                'errors': len(self.current_validation.errors),
                'warnings': len(self.current_validation.warnings),
                'quality_metrics': self.current_validation.quality_metrics
            }
        
        return summary
    
    def get_column_analysis(self) -> Dict[str, Any]:
        """
        Get detailed column analysis.
        
        Returns:
            Dictionary with column analysis information
        """
        if self.current_validation is None:
            return {}
        
        return self.current_validation.column_info
    
    def get_recommended_columns(self) -> List[str]:
        """
        Get list of columns recommended for text analysis.
        
        Returns:
            List of recommended column names
        """
        if self.current_validation is None:
            return []
        
        return self.validation_service.get_recommended_text_columns(self.current_validation)
    
    def update_column_selection(self, selected_columns: List[str]) -> Tuple[bool, str]:
        """
        Update selected columns for analysis.
        
        Args:
            selected_columns: List of column names to select
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if self.current_metadata is None:
            return False, "No data loaded"
        
        try:
            # Validate column selection
            validation_result = self.validation_service.validate_column_selection(
                selected_columns, self.current_metadata.columns
            )
            
            if not validation_result.is_valid:
                error_msg = "; ".join(validation_result.errors)
                return False, error_msg
            
            # Update data config
            self.data_config.selected_columns = selected_columns
            
            # Generate preview of combined text
            self._update_combined_text_preview()
            
            message = f"Selected {len(selected_columns)} columns for analysis"
            if validation_result.warnings:
                message += f" (with warnings: {'; '.join(validation_result.warnings)})"
            
            self._update_status(message)
            return True, message
            
        except Exception as e:
            error_msg = f"Failed to update column selection: {str(e)}"
            logger.error(f"Column selection error: {e}")
            return False, error_msg
    
    def update_text_combination_settings(self, 
                                       method: str = "concatenate",
                                       separator: str = " ",
                                       include_column_names: bool = False) -> bool:
        """
        Update text combination settings.
        
        Args:
            method: Combination method
            separator: Text separator
            include_column_names: Whether to include column names in combined text
            
        Returns:
            True if successful
        """
        try:
            self.data_config.text_combination_method = method
            self.data_config.text_combination_separator = separator
            self.data_config.include_column_names = include_column_names
            
            # Update preview
            self._update_combined_text_preview()
            
            return True
        except Exception as e:
            logger.error(f"Failed to update text combination settings: {e}")
            return False
    
    def _update_combined_text_preview(self):
        """Update the combined text preview in data config."""
        if self.current_metadata and self.current_metadata.preview_data is not None:
            # This will trigger the combined_text_preview property calculation
            preview = self.data_config.combined_text_preview
            logger.debug("Updated combined text preview")
    
    def get_combined_text_preview(self) -> str:
        """
        Get preview of how selected columns will be combined.
        
        Returns:
            String preview of combined text
        """
        return self.data_config.combined_text_preview
    
    def get_validation_issues(self) -> Tuple[List[str], List[str]]:
        """
        Get validation errors and warnings.
        
        Returns:
            Tuple of (errors: List[str], warnings: List[str])
        """
        if self.current_validation is None:
            return [], []
        
        return self.current_validation.errors, self.current_validation.warnings
    
    def is_ready_for_analysis(self) -> bool:
        """
        Check if data is ready for topic modeling analysis.
        
        Returns:
            True if data is configured and ready
        """
        return (self.data_config.is_configured and 
                self.current_validation is not None and 
                self.current_validation.is_valid)
    
    def get_analysis_ready_data(self) -> Optional[pd.Series]:
        """
        Get the combined text data ready for analysis.
        
        Returns:
            Series with combined text or None if not ready
        """
        if not self.is_ready_for_analysis() or self.current_data is None:
            return None
        
        try:
            # Combine selected columns according to configuration
            combined_texts = []
            
            for _, row in self.current_data.iterrows():
                texts = []
                for col in self.data_config.selected_columns:
                    if col in row and pd.notna(row[col]):
                        text = str(row[col]).strip()
                        if text:
                            if self.data_config.include_column_names:
                                texts.append(f"{col}: {text}")
                            else:
                                texts.append(text)
                
                if texts:
                    combined_text = self.data_config.text_combination_separator.join(texts)
                    
                    # Apply text length filters
                    if len(combined_text) >= self.data_config.min_text_length:
                        if (self.data_config.max_text_length is None or 
                            len(combined_text) <= self.data_config.max_text_length):
                            combined_texts.append(combined_text)
            
            # Filter empty rows if configured
            if self.data_config.remove_empty_rows:
                combined_texts = [text for text in combined_texts if text.strip()]
            
            return pd.Series(combined_texts)
            
        except Exception as e:
            logger.error(f"Failed to generate analysis-ready data: {e}")
            return None
    
    def get_combined_texts(self) -> List[str]:
        """Get the combined text data as a list of strings.
        
        Returns:
            List of combined text strings, or empty list if not ready
        """
        series = self.get_analysis_ready_data()
        if series is not None:
            return series.tolist()
        return []
    
    def clear_data(self):
        """Clear all loaded data and reset state."""
        self.current_data = None
        self.current_metadata = None
        self.current_validation = None
        self.data_config = DataConfig()
        
        self._update_status("Data cleared")
        logger.info("Data controller state cleared") 