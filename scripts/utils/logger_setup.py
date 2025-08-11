#!/usr/bin/env python3
"""
Logger Setup for S2 Data Pipeline
Configures logging for the pipeline components.

Author: SeeSense Data Pipeline
"""

import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime


def setup_logger(name, log_config=None, logs_dir=None):
    """
    Set up a logger with file and console handlers.
    
    Args:
        name (str): Logger name
        log_config (dict): Logging configuration
        logs_dir (str): Directory for log files
    
    Returns:
        logging.Logger: Configured logger
    """
    # Default configuration
    if log_config is None:
        log_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'max_file_size_mb': 10,
            'backup_count': 5
        }
    
    # Set up logs directory
    if logs_dir is None:
        project_root = Path(__file__).parent.parent.parent
        logs_dir = project_root / 'logs'
    else:
        logs_dir = Path(logs_dir)
    
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_config.get('level', 'INFO').upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    log_file = logs_dir / f'{name}.log'
    max_bytes = log_config.get('max_file_size_mb', 10) * 1024 * 1024
    backup_count = log_config.get('backup_count', 5)
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(getattr(logging, log_config.get('level', 'INFO').upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def quick_logger(name='pipeline'):
    """
    Quick logger setup for testing and utilities.
    
    Args:
        name (str): Logger name
    
    Returns:
        logging.Logger: Basic configured logger
    """
    return setup_logger(name)


class PipelineLogger:
    """Context manager for pipeline logging."""
    
    def __init__(self, step_name, config=None):
        """
        Initialize pipeline logger.
        
        Args:
            step_name (str): Name of the pipeline step
            config (dict): Logging configuration
        """
        self.step_name = step_name
        self.logger = setup_logger(f'pipeline_{step_name}', config)
        self.start_time = None
    
    def __enter__(self):
        """Enter context manager."""
        self.start_time = datetime.now()
        self.logger.info(f"🚀 Starting {self.step_name}")
        self.logger.info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        if exc_type is None:
            self.logger.info(f"✅ Completed {self.step_name}")
        else:
            self.logger.error(f"❌ Failed {self.step_name}: {exc_val}")
        
        self.logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Duration: {duration}")
        
        return False  # Don't suppress exceptions
