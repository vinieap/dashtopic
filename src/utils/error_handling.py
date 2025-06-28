"""
Error handling utilities and custom exception classes.
"""
import logging
import traceback
from typing import Optional

logger = logging.getLogger(__name__)

class BERTopicAppException(Exception):
    """Base exception for the application."""
    pass

class DataValidationError(BERTopicAppException):
    """Raised when data validation fails."""
    pass

class ModelConfigurationError(BERTopicAppException):
    """Raised when model configuration is invalid.""" 
    pass

class CacheError(BERTopicAppException):
    """Raised when cache operations fail."""
    pass

class ExportError(BERTopicAppException):
    """Raised when export operations fail."""
    pass

class EmbeddingError(BERTopicAppException):
    """Raised when embedding generation fails."""
    pass

def handle_exception(exception: Exception, context: Optional[str] = None) -> str:
    """
    Handle exceptions consistently across the application.
    
    Args:
        exception: The exception to handle
        context: Optional context information
        
    Returns:
        User-friendly error message
    """
    error_msg = str(exception)
    
    if context:
        error_msg = f"{context}: {error_msg}"
    
    # Log the full traceback
    logger.error(f"Exception occurred: {error_msg}")
    logger.error(traceback.format_exc())
    
    # Return user-friendly message
    if isinstance(exception, DataValidationError):
        return f"Data validation error: {str(exception)}"
    elif isinstance(exception, ModelConfigurationError):
        return f"Model configuration error: {str(exception)}"
    elif isinstance(exception, CacheError):
        return f"Cache operation failed: {str(exception)}"
    elif isinstance(exception, ExportError):
        return f"Export operation failed: {str(exception)}"
    elif isinstance(exception, EmbeddingError):
        return f"Embedding generation failed: {str(exception)}"
    elif isinstance(exception, FileNotFoundError):
        return f"File not found: {str(exception)}"
    elif isinstance(exception, PermissionError):
        return f"Permission denied: {str(exception)}"
    elif isinstance(exception, MemoryError):
        return "Insufficient memory to complete the operation. Try with a smaller dataset."
    else:
        return f"An unexpected error occurred: {str(exception)}" 