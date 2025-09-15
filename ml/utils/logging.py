"""Logging configuration"""

import logging
import sys
from pathlib import Path

def setup_logging(level=logging.INFO, log_file=None):
    """
    Thiết lập logging cho project
    
    Args:
        level: Logging level (default: INFO)
        log_file: Đường dẫn file log (optional)
    """
    # Tạo formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler nếu có log_file
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger

def get_logger(name):
    """Lấy logger cho module cụ thể"""
    return logging.getLogger(name)