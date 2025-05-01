#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logging utilities for medical LLM fine-tuning.
"""

import logging
import os
import sys
from datetime import datetime


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger with the specified name and level.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Set formatter for handlers
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger if they don't exist already
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger


def setup_file_logging(logger: logging.Logger, log_dir: str, name: str = None) -> logging.Logger:
    """
    Add file logging to an existing logger.
    
    Args:
        logger: Existing logger
        log_dir: Directory to save log files
        name: Custom name for log file (default: logger name)
        
    Returns:
        Logger with file handler added
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = name or logger.name.split(".")[-1]
    log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logger.level)
    
    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Set formatter for handler
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    logger.info(f"Logging to file: {log_file}")
    
    return logger


def log_system_info(logger: logging.Logger) -> None:
    """
    Log system information for debugging.
    
    Args:
        logger: Logger to use
    """
    try:
        import platform
        import torch
        
        logger.info(f"System: {platform.system()} {platform.release()}")
        logger.info(f"Python version: {platform.python_version()}")
        
        if torch.cuda.is_available():
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.info("CUDA not available")
    except Exception as e:
        logger.warning(f"Failed to log system info: {e}")


if __name__ == "__main__":
    # Example usage
    logger = get_logger("test_logger")
    logger = setup_file_logging(logger, "logs")
    
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    log_system_info(logger)
