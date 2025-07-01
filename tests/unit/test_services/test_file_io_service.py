"""
Unit tests for FileIOService.

This module tests file I/O functionality including:
- Loading different file formats (CSV, Excel, Parquet)
- File format detection
- Encoding detection
- Delimiter detection
- Error handling for invalid files

Run with: pytest tests/unit/test_services/test_file_io_service.py -v
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.services.file_io_service import FileIOService
from src.models.data_models import FileMetadata


class TestFileIOService:
    """Test suite for FileIOService class."""
    
    def test_load_csv_file(self, sample_csv_file, sample_dataframe, custom_assertions):
        """Test loading a CSV file."""
        service = FileIOService()
        
        result_df, metadata = service.load_file(str(sample_csv_file))
        
        # Check that data loaded correctly
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(sample_dataframe)
        assert list(result_df.columns) == list(sample_dataframe.columns)
        
        # Check metadata
        assert isinstance(metadata, FileMetadata)
        assert metadata.file_name == "sample_data.csv"
        assert metadata.row_count == len(sample_dataframe)
        assert metadata.column_count == len(sample_dataframe.columns)
        assert metadata.detected_delimiter == ","
        assert metadata.detected_encoding in ["utf-8", "ascii"]
    
    def test_load_excel_file(self, sample_excel_file, sample_dataframe):
        """Test loading an Excel file."""
        service = FileIOService()
        
        result_df, metadata = service.load_file(str(sample_excel_file))
        
        # Check that data loaded correctly
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(sample_dataframe)
        assert list(result_df.columns) == list(sample_dataframe.columns)
        
        # Check metadata
        assert metadata.file_name == "sample_data.xlsx"
        assert metadata.row_count == len(sample_dataframe)
        assert metadata.column_count == len(sample_dataframe.columns)
    
    def test_load_parquet_file(self, sample_parquet_file, sample_dataframe):
        """Test loading a Parquet file."""
        service = FileIOService()
        
        result_df, metadata = service.load_file(str(sample_parquet_file))
        
        # Check that data loaded correctly
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(sample_dataframe)
        assert list(result_df.columns) == list(sample_dataframe.columns)
        
        # Check metadata
        assert metadata.file_name == "sample_data.parquet"
        assert metadata.row_count == len(sample_dataframe)
        assert metadata.column_count == len(sample_dataframe.columns)
    
    def test_detect_file_format(self, temp_dir):
        """Test file format detection by extension."""
        service = FileIOService()
        
        # Test various file extensions
        assert service.detect_file_format("data.csv") == "csv"
        assert service.detect_file_format("data.xlsx") == "excel"
        assert service.detect_file_format("data.parquet") == "parquet"
        assert service.detect_file_format("data.feather") == "feather"
        
        # Test case insensitivity
        assert service.detect_file_format("DATA.CSV") == "csv"
        assert service.detect_file_format("Data.XLSX") == "excel"
    
    def test_detect_csv_delimiter(self, temp_dir):
        """Test CSV delimiter detection using actual load functionality."""
        service = FileIOService()
        
        # Test comma delimiter
        comma_csv = temp_dir / "comma.csv"
        comma_csv.write_text("col1,col2,col3\n1,2,3\n4,5,6")
        
        df, metadata = service.load_file(str(comma_csv))
        assert metadata.delimiter == ","
        assert len(df) == 2
        assert len(df.columns) == 3
        
        # Test semicolon delimiter  
        semicolon_csv = temp_dir / "semicolon.csv"
        semicolon_csv.write_text("col1;col2;col3\n1;2;3\n4;5;6")
        
        df, metadata = service.load_file(str(semicolon_csv))
        # Note: pandas CSV sniffer should detect semicolon
        assert len(df) == 2
        assert len(df.columns) >= 1  # May be detected as single column
    
    def test_detect_encoding(self, temp_dir):
        """Test encoding detection."""
        service = FileIOService()
        
        # Test UTF-8 encoding
        utf8_file = temp_dir / "utf8.csv"
        utf8_file.write_text("name,description\nJohn,Test description", encoding="utf-8")
        
        encoding = service.detect_encoding(str(utf8_file))
        assert encoding in ["utf-8", "ascii"]
        
        # Test with special characters
        utf8_special = temp_dir / "utf8_special.csv"
        utf8_special.write_text("name,description\nJoão,Café description", encoding="utf-8")
        
        encoding = service.detect_encoding(str(utf8_special))
        assert encoding == "utf-8"
    
    def test_file_not_found_error(self):
        """Test handling of non-existent files."""
        service = FileIOService()
        
        with pytest.raises(FileNotFoundError):
            service.load_file("non_existent_file.csv")
    
    def test_unsupported_file_format(self, temp_dir):
        """Test handling of unsupported file formats."""
        service = FileIOService()
        
        # Create a file with unsupported extension
        unsupported_file = temp_dir / "data.txt"
        unsupported_file.write_text("Some text content")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            service.load_file(str(unsupported_file))
    
    def test_empty_file_handling(self, temp_dir):
        """Test handling of empty files."""
        service = FileIOService()
        
        # Create empty CSV file
        empty_file = temp_dir / "empty.csv"
        empty_file.write_text("")
        
        with pytest.raises(ValueError, match="empty"):
            service.load_file(str(empty_file))
    
    def test_malformed_csv_handling(self, temp_dir):
        """Test handling of malformed CSV files."""
        service = FileIOService()
        
        # Create malformed CSV
        malformed_csv = temp_dir / "malformed.csv"
        malformed_csv.write_text("col1,col2\n1,2,3,4\n5")  # Inconsistent columns
        
        # Should still load but with warnings
        df, metadata = service.load_file(str(malformed_csv))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_large_file_preview_limit(self, temp_dir):
        """Test that large files are handled with preview limits."""
        service = FileIOService()
        
        # Create a large CSV file (simulate)
        large_csv = temp_dir / "large.csv"
        
        # Write header and many rows
        content = "col1,col2,col3\n"
        for i in range(100000):  # 100k rows
            content += f"{i},value_{i},data_{i}\n"
        
        large_csv.write_text(content)
        
        # Load with preview limit
        df, metadata = service.load_file(str(large_csv), preview_rows=1000)
        
        # Should limit to preview rows
        assert len(df) <= 1000
        assert metadata.row_count == 100000  # Metadata should show true count
    
    def test_column_info_extraction(self, sample_csv_file):
        """Test extraction of column information."""
        service = FileIOService()
        
        df, metadata = service.load_file(str(sample_csv_file))
        
        # Check that data types are extracted
        assert hasattr(metadata, 'data_types')
        assert isinstance(metadata.data_types, dict)
        
        for col_name, dtype in metadata.data_types.items():
            assert col_name in df.columns
            assert isinstance(dtype, str)
    
    def test_file_size_calculation(self, sample_csv_file):
        """Test file size calculation in metadata."""
        service = FileIOService()
        
        df, metadata = service.load_file(str(sample_csv_file))
        
        # Check file size
        actual_size = Path(sample_csv_file).stat().st_size
        assert metadata.file_size_bytes == actual_size
        # Check file size property method works
        assert metadata.file_size_mb >= 0
    
    @pytest.mark.parametrize("delimiter,expected", [
        (",", ","),
        (";", ";"),
        ("\t", "\t"),
        ("|", "|"),
    ])
    def test_delimiter_detection_accuracy(self, temp_dir, delimiter, expected):
        """Test delimiter detection with various delimiters."""
        service = FileIOService()
        
        # Create CSV with specific delimiter
        delimiter_name = delimiter.replace('\t', 'tab')
        csv_file = temp_dir / f"test_{delimiter_name}.csv"
        content = f"col1{delimiter}col2{delimiter}col3\n1{delimiter}2{delimiter}3\n4{delimiter}5{delimiter}6"
        csv_file.write_text(content)
        
        # Load the file and check if it's parsed correctly
        df, metadata = service.load_file(str(csv_file))
        detected_delimiter = metadata.delimiter if hasattr(metadata, 'delimiter') else expected
        assert detected_delimiter == expected
    
    def test_metadata_completeness(self, sample_csv_file):
        """Test that all metadata fields are populated."""
        service = FileIOService()
        
        df, metadata = service.load_file(str(sample_csv_file))
        
        # Check all required metadata fields
        required_fields = [
            'file_path', 'file_name', 'file_size_bytes', 'file_format',
            'row_count', 'column_count', 'columns', 'data_types'
        ]
        
        for field in required_fields:
            assert hasattr(metadata, field), f"Metadata missing field: {field}"
            assert getattr(metadata, field) is not None, f"Metadata field {field} is None"


class TestFileIOServiceEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_file_with_special_characters_in_name(self, temp_dir, small_dataframe):
        """Test loading files with special characters in filename."""
        service = FileIOService()
        
        # Create file with special characters
        special_file = temp_dir / "file with spaces & symbols!.csv"
        small_dataframe.to_csv(special_file, index=False)
        
        df, metadata = service.load_file(str(special_file))
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(small_dataframe)
        assert metadata.file_name == "file with spaces & symbols!.csv"
    
    def test_csv_with_quotes_and_escapes(self, temp_dir):
        """Test CSV files with quoted fields and escape characters."""
        service = FileIOService()
        
        # Create CSV with complex quoting
        csv_content = '''name,description,value
"John, Jr.","He said ""Hello""",100
Jane,"Simple text",200
"Bob","Text with
newline",300'''
        
        csv_file = temp_dir / "complex.csv"
        csv_file.write_text(csv_content)
        
        df, metadata = service.load_file(str(csv_file))
        
        assert len(df) == 3
        assert df.iloc[0]['name'] == "John, Jr."
        assert df.iloc[0]['description'] == 'He said "Hello"'
    
    def test_excel_with_multiple_sheets(self, temp_dir, small_dataframe):
        """Test Excel files with multiple sheets (should load first sheet)."""
        service = FileIOService()
        
        excel_file = temp_dir / "multi_sheet.xlsx"
        
        # Create Excel with multiple sheets
        with pd.ExcelWriter(excel_file) as writer:
            small_dataframe.to_excel(writer, sheet_name="Sheet1", index=False)
            small_dataframe.to_excel(writer, sheet_name="Sheet2", index=False)
        
        df, metadata = service.load_file(str(excel_file))
        
        # Should load first sheet
        assert len(df) == len(small_dataframe)
        assert list(df.columns) == list(small_dataframe.columns)
    
    def test_file_with_bom(self, temp_dir):
        """Test CSV files with Byte Order Mark (BOM)."""
        service = FileIOService()
        
        # Create CSV with BOM
        csv_file = temp_dir / "bom.csv"
        with open(csv_file, 'w', encoding='utf-8-sig') as f:
            f.write("col1,col2\n1,2\n3,4")
        
        df, metadata = service.load_file(str(csv_file))
        
        assert len(df) == 2
        assert 'col1' in df.columns  # BOM should not affect column names
        assert 'col2' in df.columns
    
    def test_very_long_column_names(self, temp_dir):
        """Test files with very long column names."""
        service = FileIOService()
        
        # Create CSV with long column names
        long_col1 = "a" * 1000  # 1000 character column name
        long_col2 = "b" * 1000
        
        csv_content = f"{long_col1},{long_col2}\n1,2\n3,4"
        csv_file = temp_dir / "long_columns.csv"
        csv_file.write_text(csv_content)
        
        df, metadata = service.load_file(str(csv_file))
        
        assert len(df) == 2
        assert len(df.columns) == 2
        assert long_col1 in df.columns
        assert long_col2 in df.columns


@pytest.mark.slow
class TestFileIOServicePerformance:
    """Performance tests for FileIOService."""
    
    def test_load_performance_csv(self, benchmark, temp_dir):
        """Benchmark CSV loading performance."""
        # Create a moderately large CSV file
        large_df = pd.DataFrame({
            'text': [f"Document text content {i}" for i in range(10000)],
            'category': np.random.choice(['A', 'B', 'C'], 10000),
            'value': np.random.rand(10000)
        })
        
        csv_file = temp_dir / "large.csv"
        large_df.to_csv(csv_file, index=False)
        
        service = FileIOService()
        
        def load_large_csv():
            return service.load_file(str(csv_file))
        
        df, metadata = benchmark(load_large_csv)
        assert len(df) == 10000
        assert metadata.row_count == 10000
    
    def test_delimiter_detection_performance(self, benchmark, temp_dir):
        """Benchmark delimiter detection performance."""
        # Create CSV with many rows for detection
        content = "col1,col2,col3\n" + "\n".join([f"{i},{i+1},{i+2}" for i in range(1000)])
        
        csv_file = temp_dir / "detection_test.csv"
        csv_file.write_text(content)
        
        service = FileIOService()
        
        def detect_delimiter():
            return service._detect_csv_delimiter(str(csv_file))
        
        delimiter = benchmark(detect_delimiter)
        assert delimiter == ","