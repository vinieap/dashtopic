"""
Streaming Data Loader

Memory-efficient data loading for large files using chunked processing.
Reduces memory footprint by processing data in manageable chunks.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, Tuple, List, Union
import chardet
from contextlib import contextmanager

from .memory_manager import MemoryOptimizer, get_memory_monitor

logger = logging.getLogger(__name__)


class StreamingDataLoader:
    """Memory-efficient data loader for large files."""
    
    def __init__(self, chunk_size: int = 10000, memory_limit_mb: float = 1000.0):
        """Initialize the streaming loader.
        
        Args:
            chunk_size: Number of rows per chunk
            memory_limit_mb: Memory limit in MB for processing
        """
        self.chunk_size = chunk_size
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.memory_monitor = get_memory_monitor()
        
    def stream_csv(self, file_path: str, encoding: Optional[str] = None, 
                   delimiter: Optional[str] = None, **kwargs) -> Iterator[pd.DataFrame]:
        """Stream CSV file in chunks.
        
        Args:
            file_path: Path to CSV file
            encoding: File encoding (auto-detected if None)
            delimiter: CSV delimiter (auto-detected if None)
            **kwargs: Additional pandas read_csv arguments
            
        Yields:
            DataFrame chunks
        """
        if encoding is None:
            encoding = self._detect_encoding(file_path)
        
        if delimiter is None:
            delimiter = self._detect_delimiter(file_path, encoding)
        
        logger.info(f"Streaming CSV file: {file_path} (encoding={encoding}, delimiter='{delimiter}')")
        
        try:
            # Use pandas chunking functionality
            chunk_reader = pd.read_csv(
                file_path,
                encoding=encoding,
                delimiter=delimiter,
                chunksize=self.chunk_size,
                **kwargs
            )
            
            chunk_count = 0
            for chunk in chunk_reader:
                chunk_count += 1
                
                # Optimize chunk memory usage
                optimized_chunk = MemoryOptimizer.optimize_pandas_memory(chunk)
                
                # Check memory pressure
                memory_stats = self.memory_monitor.get_current_stats()
                if memory_stats.is_high_pressure:
                    logger.warning(f"High memory pressure during streaming at chunk {chunk_count}")
                    # Force garbage collection
                    MemoryOptimizer.force_garbage_collection()
                
                logger.debug(f"Yielding chunk {chunk_count}: {len(optimized_chunk)} rows")
                yield optimized_chunk
                
        except Exception as e:
            logger.error(f"Error streaming CSV file: {e}")
            raise
    
    def stream_excel(self, file_path: str, sheet_name: Optional[str] = None,
                     **kwargs) -> Iterator[pd.DataFrame]:
        """Stream Excel file in chunks.
        
        Note: Excel streaming is limited - loads full sheet then chunks.
        For very large Excel files, consider converting to CSV first.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name (first sheet if None)
            **kwargs: Additional pandas read_excel arguments
            
        Yields:
            DataFrame chunks
        """
        logger.info(f"Streaming Excel file: {file_path}")
        
        try:
            # Load full Excel sheet (limitation of pandas)
            full_df = pd.read_excel(
                file_path,
                sheet_name=sheet_name or 0,
                engine='openpyxl',
                **kwargs
            )
            
            # Optimize memory usage
            full_df = MemoryOptimizer.optimize_pandas_memory(full_df)
            
            # Yield in chunks
            for chunk_start in range(0, len(full_df), self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, len(full_df))
                chunk = full_df.iloc[chunk_start:chunk_end].copy()
                
                logger.debug(f"Yielding Excel chunk: rows {chunk_start}-{chunk_end}")
                yield chunk
            
            # Clean up full dataframe
            del full_df
            MemoryOptimizer.force_garbage_collection()
            
        except Exception as e:
            logger.error(f"Error streaming Excel file: {e}")
            raise
    
    def stream_parquet(self, file_path: str, **kwargs) -> Iterator[pd.DataFrame]:
        """Stream Parquet file in chunks.
        
        Args:
            file_path: Path to Parquet file
            **kwargs: Additional pandas read_parquet arguments
            
        Yields:
            DataFrame chunks
        """
        logger.info(f"Streaming Parquet file: {file_path}")
        
        try:
            import pyarrow.parquet as pq
            
            # Open parquet file
            parquet_file = pq.ParquetFile(file_path)
            
            # Calculate batch size based on memory limit
            # This is approximate - actual memory usage depends on data types
            estimated_batch_size = min(self.chunk_size, 
                                     max(1000, self.memory_limit_bytes // (parquet_file.schema.nbytes * 8)))
            
            # Read in batches
            for batch in parquet_file.iter_batches(batch_size=estimated_batch_size):
                chunk = batch.to_pandas()
                
                # Optimize memory usage
                chunk = MemoryOptimizer.optimize_pandas_memory(chunk)
                
                logger.debug(f"Yielding Parquet chunk: {len(chunk)} rows")
                yield chunk
                
        except ImportError:
            # Fallback to pandas if pyarrow not available
            logger.warning("PyArrow not available, falling back to pandas (less efficient)")
            full_df = pd.read_parquet(file_path, **kwargs)
            full_df = MemoryOptimizer.optimize_pandas_memory(full_df)
            
            for chunk_start in range(0, len(full_df), self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, len(full_df))
                chunk = full_df.iloc[chunk_start:chunk_end].copy()
                yield chunk
            
            del full_df
            MemoryOptimizer.force_garbage_collection()
            
        except Exception as e:
            logger.error(f"Error streaming Parquet file: {e}")
            raise
    
    def stream_file(self, file_path: str, **kwargs) -> Iterator[pd.DataFrame]:
        """Stream file based on format detection.
        
        Args:
            file_path: Path to file
            **kwargs: Format-specific arguments
            
        Yields:
            DataFrame chunks
        """
        file_format = self._detect_file_format(file_path)
        
        if file_format == 'csv':
            yield from self.stream_csv(file_path, **kwargs)
        elif file_format == 'excel':
            yield from self.stream_excel(file_path, **kwargs)
        elif file_format == 'parquet':
            yield from self.stream_parquet(file_path, **kwargs)
        elif file_format == 'feather':
            # Feather doesn't have native streaming, load and chunk
            full_df = pd.read_feather(file_path)
            full_df = MemoryOptimizer.optimize_pandas_memory(full_df)
            
            for chunk_start in range(0, len(full_df), self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, len(full_df))
                chunk = full_df.iloc[chunk_start:chunk_end].copy()
                yield chunk
            
            del full_df
            MemoryOptimizer.force_garbage_collection()
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    
    def load_sample(self, file_path: str, sample_size: int = 1000, **kwargs) -> pd.DataFrame:
        """Load a sample of the file for preview/analysis.
        
        Args:
            file_path: Path to file
            sample_size: Number of rows to sample
            **kwargs: Format-specific arguments
            
        Returns:
            Sample DataFrame
        """
        logger.info(f"Loading sample from {file_path} ({sample_size} rows)")
        
        file_format = self._detect_file_format(file_path)
        
        if file_format == 'csv':
            encoding = kwargs.get('encoding') or self._detect_encoding(file_path)
            delimiter = kwargs.get('delimiter') or self._detect_delimiter(file_path, encoding)
            
            sample_df = pd.read_csv(
                file_path,
                encoding=encoding,
                delimiter=delimiter,
                nrows=sample_size
            )
        elif file_format == 'excel':
            sample_df = pd.read_excel(
                file_path,
                sheet_name=kwargs.get('sheet_name', 0),
                nrows=sample_size,
                engine='openpyxl'
            )
        elif file_format == 'parquet':
            # For parquet, read first chunk
            chunk_iter = self.stream_parquet(file_path, **kwargs)
            sample_df = next(chunk_iter).head(sample_size)
        elif file_format == 'feather':
            # Feather doesn't support nrows, so load and sample
            full_df = pd.read_feather(file_path)
            sample_df = full_df.head(sample_size)
            del full_df
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        # Optimize sample memory
        sample_df = MemoryOptimizer.optimize_pandas_memory(sample_df)
        
        logger.info(f"Loaded sample: {len(sample_df)} rows, {len(sample_df.columns)} columns")
        return sample_df
    
    def get_file_info_streaming(self, file_path: str) -> Dict[str, Any]:
        """Get file information without loading the full file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        path = Path(file_path)
        file_size = path.stat().st_size
        file_format = self._detect_file_format(file_path)
        
        # Get row/column counts efficiently
        row_count = 0
        column_count = 0
        
        try:
            if file_format == 'csv':
                # Count lines efficiently
                with open(file_path, 'rb') as f:
                    row_count = sum(1 for _ in f) - 1  # Subtract header
                
                # Get column count from first row
                sample = self.load_sample(file_path, sample_size=1)
                column_count = len(sample.columns)
                
            elif file_format == 'parquet':
                try:
                    import pyarrow.parquet as pq
                    parquet_file = pq.ParquetFile(file_path)
                    row_count = parquet_file.metadata.num_rows
                    column_count = parquet_file.schema.nbytes
                except ImportError:
                    # Fallback to sampling
                    sample = self.load_sample(file_path, sample_size=1)
                    column_count = len(sample.columns)
                    # Estimate row count (not exact)
                    row_count = -1
            
            else:
                # For Excel/Feather, use sampling approach
                sample = self.load_sample(file_path, sample_size=1)
                column_count = len(sample.columns)
                # Row count estimation would require full load
                row_count = -1
        
        except Exception as e:
            logger.warning(f"Could not determine file dimensions: {e}")
            row_count = -1
            column_count = -1
        
        return {
            'file_name': path.name,
            'file_size_bytes': file_size,
            'file_size_mb': file_size / (1024 * 1024),
            'file_format': file_format,
            'estimated_row_count': row_count,
            'column_count': column_count,
            'supports_streaming': file_format in ['csv', 'parquet']
        }
    
    @contextmanager
    def memory_limited_processing(self):
        """Context manager for memory-limited processing."""
        initial_stats = self.memory_monitor.get_current_stats()
        
        try:
            yield
        finally:
            # Cleanup after processing
            final_stats = self.memory_monitor.get_current_stats()
            memory_increase = final_stats.process_mb - initial_stats.process_mb
            
            if memory_increase > 100:  # If we used more than 100MB
                logger.info(f"Memory increased by {memory_increase:.1f}MB during processing")
                MemoryOptimizer.force_garbage_collection()
    
    def _detect_file_format(self, file_path: str) -> str:
        """Detect file format from extension."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        format_map = {
            '.csv': 'csv',
            '.xlsx': 'excel',
            '.parquet': 'parquet',
            '.feather': 'feather'
        }
        
        if extension not in format_map:
            raise ValueError(f"Unsupported file format: {extension}")
        
        return format_map[extension]
    
    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding for text files."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                if confidence < 0.7 or encoding.lower() == 'ascii':
                    encoding = 'utf-8'
                
                return encoding
        except Exception:
            return 'utf-8'
    
    def _detect_delimiter(self, file_path: str, encoding: str) -> str:
        """Detect CSV delimiter."""
        delimiters = [',', ';', '\t', '|']
        
        for delimiter in delimiters:
            try:
                sample_df = pd.read_csv(
                    file_path,
                    delimiter=delimiter,
                    encoding=encoding,
                    nrows=5
                )
                
                if len(sample_df.columns) > 1:
                    return delimiter
            except Exception:
                continue
        
        return ','  # Default fallback


class ChunkedProcessor:
    """Process data in chunks with memory management."""
    
    def __init__(self, streaming_loader: StreamingDataLoader):
        self.streaming_loader = streaming_loader
        
    def process_file_in_chunks(self, file_path: str, 
                             processor_func: callable,
                             combine_func: callable = None,
                             **kwargs) -> Any:
        """Process a file in chunks.
        
        Args:
            file_path: Path to file
            processor_func: Function to apply to each chunk
            combine_func: Function to combine results (if None, returns list)
            **kwargs: Arguments for streaming loader
            
        Returns:
            Combined processing results
        """
        results = []
        
        with self.streaming_loader.memory_limited_processing():
            for chunk_idx, chunk in enumerate(self.streaming_loader.stream_file(file_path, **kwargs)):
                try:
                    logger.debug(f"Processing chunk {chunk_idx + 1}")
                    
                    # Process the chunk
                    chunk_result = processor_func(chunk)
                    results.append(chunk_result)
                    
                    # Clear chunk from memory
                    del chunk
                    
                    # Periodic cleanup
                    if (chunk_idx + 1) % 10 == 0:
                        MemoryOptimizer.force_garbage_collection()
                        
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_idx + 1}: {e}")
                    raise
        
        # Combine results
        if combine_func:
            return combine_func(results)
        else:
            return results
    
    def extract_text_chunked(self, file_path: str, text_columns: List[str],
                           separator: str = " ") -> Iterator[List[str]]:
        """Extract text from file in chunks.
        
        Args:
            file_path: Path to file
            text_columns: Columns to extract text from
            separator: Text combination separator
            
        Yields:
            Lists of extracted text strings
        """
        for chunk in self.streaming_loader.stream_file(file_path):
            # Extract text from chunk
            text_list = []
            for _, row in chunk.iterrows():
                text_parts = [str(row[col]) for col in text_columns if pd.notna(row[col])]
                combined_text = separator.join(text_parts)
                text_list.append(combined_text)
            
            yield text_list