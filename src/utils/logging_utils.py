"""
Logging configuration utilities.
"""
import logging
import os
from pathlib import Path

def setup_logging(log_level=logging.INFO):
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: The logging level (default: logging.INFO)
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "app.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create separate error logger
    error_logger = logging.getLogger('error')
    error_handler = logging.FileHandler(log_dir / "error.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    error_logger.addHandler(error_handler)
    
    # Create performance logger
    perf_logger = logging.getLogger('performance')
    perf_handler = logging.FileHandler(log_dir / "performance.log")
    perf_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(message)s')
    )
    perf_logger.addHandler(perf_handler)
    
    logging.info("Logging system initialized") 