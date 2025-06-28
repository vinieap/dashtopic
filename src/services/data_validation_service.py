"""
Data validation service for quality checks and column analysis.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import re

from ..models.data_models import ValidationResult, FileMetadata, DataQualityMetrics

logger = logging.getLogger(__name__)


class DataValidationService:
    """Service for validating data quality and performing data analysis."""
    
    def __init__(self):
        self.text_patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'url': r'^https?://[^\s/$.?#].[^\s]*$',
            'phone': r'^\+?[\d\s\-\(\)]{7,}$',
            'date': r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$'
        }
    
    def validate_file_data(self, df: pd.DataFrame, metadata: FileMetadata) -> ValidationResult:
        """
        Perform comprehensive validation on loaded data.
        
        Args:
            df: Loaded DataFrame
            metadata: File metadata
            
        Returns:
            ValidationResult with errors, warnings, and metrics
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # Basic validation checks
            self._validate_data_structure(df, result)
            self._validate_data_content(df, result)
            self._analyze_columns(df, result)
            
            # Generate quality metrics
            quality_metrics = self._calculate_quality_metrics(df)
            result.quality_metrics = quality_metrics.__dict__
            
            logger.info(f"Data validation completed. Valid: {result.is_valid}, "
                       f"Errors: {len(result.errors)}, Warnings: {len(result.warnings)}")
            
        except Exception as e:
            result.add_error(f"Validation failed: {str(e)}")
            logger.error(f"Data validation error: {e}")
        
        return result
    
    def _validate_data_structure(self, df: pd.DataFrame, result: ValidationResult):
        """Validate basic data structure."""
        # Check if DataFrame is empty
        if df.empty:
            result.add_error("Dataset is empty")
            return
        
        # Check minimum size requirements
        if len(df) < 10:
            result.add_warning(f"Dataset has only {len(df)} rows. Consider using more data for better results.")
        
        if len(df.columns) < 1:
            result.add_error("Dataset has no columns")
            return
        
        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            duplicate_cols = [col for col in df.columns if list(df.columns).count(col) > 1]
            result.add_error(f"Duplicate column names found: {duplicate_cols}")
        
        # Check for unnamed columns
        unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
        if unnamed_cols:
            result.add_warning(f"Found {len(unnamed_cols)} unnamed columns. Consider using files with proper headers.")
    
    def _validate_data_content(self, df: pd.DataFrame, result: ValidationResult):
        """Validate data content quality."""
        # Check for completely empty rows
        empty_rows = df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            result.add_warning(f"Found {empty_rows} completely empty rows")
        
        # Check for columns with all missing values
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            result.add_warning(f"Columns with all missing values: {empty_columns}")
        
        # Check for columns with mostly missing values (>90%)
        high_missing_cols = []
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct > 0.9:
                high_missing_cols.append(f"{col} ({missing_pct*100:.1f}% missing)")
        
        if high_missing_cols:
            result.add_warning(f"Columns with >90% missing values: {high_missing_cols}")
        
        # Check for potential encoding issues
        text_columns = df.select_dtypes(include=['object']).columns
        encoding_issues = 0
        for col in text_columns:
            sample_values = df[col].dropna().astype(str).head(100)
            for value in sample_values:
                if any(char in value for char in ['�', '\ufffd']):
                    encoding_issues += 1
                    break
        
        if encoding_issues > 0:
            result.add_warning(f"Potential character encoding issues detected in {encoding_issues} columns")
    
    def _analyze_columns(self, df: pd.DataFrame, result: ValidationResult):
        """Analyze individual columns and detect data patterns."""
        column_info = {}
        
        for col in df.columns:
            col_analysis = self._analyze_single_column(df[col])
            column_info[col] = col_analysis
            
            # Add warnings for problematic columns
            if col_analysis['data_type'] == 'object' and col_analysis['unique_ratio'] > 0.95:
                result.add_warning(f"Column '{col}' has very high uniqueness ({col_analysis['unique_ratio']:.1%}), "
                                 "may not be suitable for text analysis")
            
            if col_analysis['missing_percentage'] > 50:
                result.add_warning(f"Column '{col}' has {col_analysis['missing_percentage']:.1f}% missing values")
        
        result.column_info = column_info
    
    def _analyze_single_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze a single column and return detailed information."""
        analysis = {
            'name': series.name,
            'data_type': str(series.dtype),
            'total_values': len(series),
            'missing_values': series.isnull().sum(),
            'missing_percentage': (series.isnull().sum() / len(series)) * 100,
            'unique_values': series.nunique(),
            'unique_ratio': series.nunique() / len(series) if len(series) > 0 else 0
        }
        
        # Type-specific analysis
        if series.dtype == 'object':
            analysis.update(self._analyze_text_column(series))
        elif pd.api.types.is_numeric_dtype(series):
            analysis.update(self._analyze_numeric_column(series))
        elif pd.api.types.is_datetime64_any_dtype(series):
            analysis.update(self._analyze_datetime_column(series))
        
        return analysis
    
    def _analyze_text_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze text column specifics."""
        non_null_series = series.dropna().astype(str)
        
        if len(non_null_series) == 0:
            return {
                'suggested_for_analysis': False,
                'average_length': 0,
                'min_length': 0,
                'max_length': 0,
                'contains_pattern': None
            }
        
        # Calculate text statistics
        lengths = non_null_series.str.len()
        
        analysis = {
            'suggested_for_analysis': True,
            'average_length': lengths.mean(),
            'min_length': lengths.min(),
            'max_length': lengths.max(),
            'contains_pattern': None
        }
        
        # Check for common patterns
        sample_values = non_null_series.head(100)
        for pattern_name, pattern in self.text_patterns.items():
            matches = sum(1 for value in sample_values if re.match(pattern, str(value)))
            if matches > len(sample_values) * 0.8:  # 80% match threshold
                analysis['contains_pattern'] = pattern_name
                break
        
        # Determine if suitable for text analysis
        avg_length = analysis['average_length']
        unique_ratio = analysis.get('unique_ratio', 0)
        
        # Not suitable if: too short, too repetitive, or contains structured data
        if avg_length < 10:
            analysis['suggested_for_analysis'] = False
        elif unique_ratio < 0.1:  # Too repetitive
            analysis['suggested_for_analysis'] = False
        elif analysis['contains_pattern'] in ['email', 'url', 'phone']:
            analysis['suggested_for_analysis'] = False
        
        return analysis
    
    def _analyze_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze numeric column specifics."""
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return {'suggested_for_analysis': False}
        
        return {
            'suggested_for_analysis': False,  # Numeric columns not suitable for text analysis
            'min_value': non_null_series.min(),
            'max_value': non_null_series.max(),
            'mean_value': non_null_series.mean(),
            'std_deviation': non_null_series.std()
        }
    
    def _analyze_datetime_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze datetime column specifics."""
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return {'suggested_for_analysis': False}
        
        return {
            'suggested_for_analysis': False,  # Datetime columns not suitable for text analysis
            'min_date': non_null_series.min(),
            'max_date': non_null_series.max(),
            'date_range_days': (non_null_series.max() - non_null_series.min()).days
        }
    
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> DataQualityMetrics:
        """Calculate comprehensive data quality metrics."""
        total_rows = len(df)
        total_cells = df.size
        
        # Count empty rows (all values are null)
        empty_rows = df.isnull().all(axis=1).sum()
        
        # Count duplicate rows
        duplicate_rows = df.duplicated().sum()
        
        # Count missing values per column
        columns_with_missing = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                columns_with_missing[col] = missing_count
        
        # Calculate text length statistics for text columns
        text_columns = df.select_dtypes(include=['object']).columns
        text_length_stats = {}
        
        if len(text_columns) > 0:
            all_text_lengths = []
            for col in text_columns:
                lengths = df[col].dropna().astype(str).str.len()
                all_text_lengths.extend(lengths.tolist())
            
            if all_text_lengths:
                text_length_stats = {
                    'min': min(all_text_lengths),
                    'max': max(all_text_lengths),
                    'mean': np.mean(all_text_lengths),
                    'median': np.median(all_text_lengths)
                }
        
        # Calculate data completeness (non-null cells / total cells)
        non_null_cells = df.count().sum()
        data_completeness = non_null_cells / total_cells if total_cells > 0 else 0
        
        # Character encoding issues (rough estimate)
        encoding_issues = 0
        for col in text_columns:
            sample_values = df[col].dropna().astype(str).head(100)
            for value in sample_values:
                if any(char in value for char in ['�', '\ufffd']):
                    encoding_issues += 1
        
        return DataQualityMetrics(
            total_rows=total_rows,
            empty_rows=empty_rows,
            duplicate_rows=duplicate_rows,
            columns_with_missing=columns_with_missing,
            text_length_stats=text_length_stats,
            character_encoding_issues=encoding_issues,
            data_completeness=data_completeness
        )
    
    def get_recommended_text_columns(self, validation_result: ValidationResult) -> List[str]:
        """
        Get list of columns recommended for text analysis.
        
        Args:
            validation_result: Result from validate_file_data
            
        Returns:
            List of column names suitable for text analysis
        """
        recommended_columns = []
        
        for col_name, col_info in validation_result.column_info.items():
            if col_info.get('suggested_for_analysis', False):
                recommended_columns.append(col_name)
        
        return recommended_columns
    
    def validate_column_selection(self, columns: List[str], available_columns: List[str]) -> ValidationResult:
        """
        Validate that selected columns are appropriate for text analysis.
        
        Args:
            columns: List of selected column names
            available_columns: List of all available columns
            
        Returns:
            ValidationResult with any issues found
        """
        result = ValidationResult(is_valid=True)
        
        if not columns:
            result.add_error("No columns selected for analysis")
            return result
        
        # Check if all selected columns exist
        missing_columns = [col for col in columns if col not in available_columns]
        if missing_columns:
            result.add_error(f"Selected columns not found: {missing_columns}")
        
        # Check for reasonable number of columns
        if len(columns) > 10:
            result.add_warning("Large number of columns selected. This may result in very long combined text.")
        
        return result 