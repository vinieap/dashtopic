"""
File I/O service for loading and managing different data formats.
"""
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import chardet
import os

from ..models.data_models import FileMetadata
from ..utils.constants import SUPPORTED_FORMATS, MAX_PREVIEW_ROWS

logger = logging.getLogger(__name__)


class FileIOService:
    """Service for handling file input/output operations."""
    
    def __init__(self):
        self.supported_formats = SUPPORTED_FORMATS
    
    def detect_file_format(self, file_path: str) -> str:
        """
        Detect file format from file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File format string (csv, xlsx, parquet, feather)
            
        Raises:
            ValueError: If file format is not supported
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {extension}. "
                           f"Supported formats: {', '.join(self.supported_formats)}")
        
        # Map extensions to format names
        format_map = {
            '.csv': 'csv',
            '.xlsx': 'excel',
            '.parquet': 'parquet', 
            '.feather': 'feather'
        }
        
        return format_map[extension]
    
    def detect_encoding(self, file_path: str) -> str:
        """
        Detect file encoding for text-based files.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected encoding string
        """
        try:
            # Read a sample of the file for encoding detection
            with open(file_path, 'rb') as file:
                raw_data = file.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                # Use utf-8 if confidence is low or if ascii is detected (often incorrect)
                if confidence < 0.7 or encoding.lower() == 'ascii':
                    encoding = 'utf-8'
                    
                logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
                return encoding
                
        except Exception as e:
            logger.warning(f"Failed to detect encoding: {e}. Using utf-8.")
            return 'utf-8'
    
    def load_csv_file(self, file_path: str, encoding: Optional[str] = None) -> pd.DataFrame:
        """
        Load CSV file with automatic delimiter and encoding detection.
        
        Args:
            file_path: Path to CSV file
            encoding: File encoding (auto-detected if None)
            
        Returns:
            Loaded DataFrame
        """
        if encoding is None:
            encoding = self.detect_encoding(file_path)
        
        # Try common delimiters
        delimiters = [',', ';', '\t', '|']
        
        for delimiter in delimiters:
            try:
                # Try to read first few rows to detect delimiter
                sample_df = pd.read_csv(
                    file_path, 
                    delimiter=delimiter, 
                    encoding=encoding,
                    nrows=5
                )
                
                # Check if we have reasonable number of columns
                if len(sample_df.columns) > 1:
                    # Load full file
                    df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
                    logger.info(f"CSV loaded successfully with delimiter '{delimiter}'")
                    return df
                    
            except Exception as e:
                logger.debug(f"Failed to load CSV with delimiter '{delimiter}': {e}")
                continue
        
        # Fallback to pandas automatic detection
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            logger.info("CSV loaded with automatic delimiter detection")
            return df
        except UnicodeDecodeError:
            # Try UTF-8 if the detected encoding fails
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                logger.info("CSV loaded with UTF-8 fallback encoding")
                return df
            except Exception as e:
                raise ValueError(f"Failed to load CSV file with multiple encodings: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load CSV file: {e}")
    
    def load_excel_file(self, file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load Excel file.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name to load (first sheet if None)
            
        Returns:
            Loaded DataFrame
        """
        try:
            # Load first sheet if no sheet specified
            df = pd.read_excel(file_path, sheet_name=sheet_name or 0, engine='openpyxl')
            logger.info(f"Excel file loaded successfully")
            return df
        except Exception as e:
            raise ValueError(f"Failed to load Excel file: {e}")
    
    def load_parquet_file(self, file_path: str) -> pd.DataFrame:
        """
        Load Parquet file.
        
        Args:
            file_path: Path to Parquet file
            
        Returns:
            Loaded DataFrame
        """
        try:
            df = pd.read_parquet(file_path, engine='pyarrow')
            logger.info("Parquet file loaded successfully")
            return df
        except Exception as e:
            raise ValueError(f"Failed to load Parquet file: {e}")
    
    def load_feather_file(self, file_path: str) -> pd.DataFrame:
        """
        Load Feather file.
        
        Args:
            file_path: Path to Feather file
            
        Returns:
            Loaded DataFrame
        """
        try:
            df = pd.read_feather(file_path)
            logger.info("Feather file loaded successfully")
            return df
        except Exception as e:
            raise ValueError(f"Failed to load Feather file: {e}")
    
    def load_file(self, file_path: str, **kwargs) -> Tuple[pd.DataFrame, FileMetadata]:
        """
        Load file and create metadata.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional arguments for specific loaders
            
        Returns:
            Tuple of (DataFrame, FileMetadata)
            
        Raises:
            ValueError: If file cannot be loaded
        """
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
        
        # Detect file format
        file_format = self.detect_file_format(file_path)
        
        # Load data based on format
        if file_format == 'csv':
            df = self.load_csv_file(file_path, kwargs.get('encoding'))
        elif file_format == 'excel':
            df = self.load_excel_file(file_path, kwargs.get('sheet_name'))
        elif file_format == 'parquet':
            df = self.load_parquet_file(file_path)
        elif file_format == 'feather':
            df = self.load_feather_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        # Create file metadata
        metadata = self._create_file_metadata(file_path, df, file_format)
        
        logger.info(f"File loaded successfully: {metadata.file_name} "
                   f"({metadata.row_count} rows, {metadata.column_count} columns)")
        
        return df, metadata
    
    def _create_file_metadata(self, file_path: str, df: pd.DataFrame, file_format: str) -> FileMetadata:
        """
        Create file metadata object.
        
        Args:
            file_path: Path to the file
            df: Loaded DataFrame
            file_format: Detected file format
            
        Returns:
            FileMetadata object
        """
        path = Path(file_path)
        
        # Get data types
        data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Create preview data (first N rows)
        preview_data = df.head(MAX_PREVIEW_ROWS).copy()
        
        # Get encoding for text files
        encoding = None
        if file_format == 'csv':
            encoding = self.detect_encoding(file_path)
        
        metadata = FileMetadata(
            file_path=file_path,
            file_name=path.name,
            file_size_bytes=path.stat().st_size,
            file_format=file_format,
            row_count=len(df),
            column_count=len(df.columns),
            columns=list(df.columns),
            data_types=data_types,
            encoding=encoding,
            preview_data=preview_data
        )
        
        return metadata
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get basic file information without loading the full file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
        
        path = Path(file_path)
        file_size = path.stat().st_size
        
        try:
            file_format = self.detect_file_format(file_path)
        except ValueError as e:
            file_format = "unsupported"
        
        return {
            'file_name': path.name,
            'file_size_bytes': file_size,
            'file_size_mb': file_size / (1024 * 1024),
            'file_format': file_format,
            'supported': file_format != "unsupported"
        } 