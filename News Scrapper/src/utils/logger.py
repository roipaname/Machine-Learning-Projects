"""
Centralized logging configuration.

Provides consistent logging across all modules with:
- Multiple output handlers (console, file, error file)
- Log rotation
- Structured logging
- Context management
"""

import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
from loguru import logger

from config.settings import (
    LOG_LEVEL,
    LOG_FILE,
    ERROR_LOG_FILE,
    LOG_ROTATION,
    LOG_RETENTION,
    LOG_FORMAT
)


def setup_logger(
    name: Optional[str] = None,
    level: str = LOG_LEVEL,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logger:
    """
    Configure application-wide logging.
    
    Args:
        name: Logger name (for filtering)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Enable file logging
        log_to_console: Enable console logging
        
    Returns:
        Configured logger instance
    """
    # Remove default handler
    logger.remove()
    
    # Console handler
    if log_to_console:
        logger.add(
            sys.stderr,
            format=LOG_FORMAT,
            level=level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
    
    # File handler for all logs
    if log_to_file:
        log_path = Path(LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path,
            format=LOG_FORMAT,
            level=level,
            rotation=LOG_ROTATION,
            retention=LOG_RETENTION,
            compression="zip",
            backtrace=True,
            diagnose=True
        )
    
    # Separate file for errors
    if log_to_file:
        error_path = Path(ERROR_LOG_FILE)
        error_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            error_path,
            format=LOG_FORMAT,
            level="ERROR",
            rotation=LOG_ROTATION,
            retention=LOG_RETENTION,
            compression="zip",
            backtrace=True,
            diagnose=True
        )
    
    logger.info(f"Logger initialized (level={level})")
    
    return logger


def log_execution_time(func):
    """
    Decorator to log function execution time.
    
    Usage:
        @log_execution_time
        def my_function():
            pass
    """
    from functools import wraps
    import time
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = func.__name__
        
        logger.debug(f"Starting {func_name}")
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func_name} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func_name} failed after {elapsed:.2f}s: {e}")
            raise
    
    return wrapper


class LogContext:
    """
    Context manager for structured logging.
    
    Usage:
        with LogContext("scraping", source="BBC"):
            scrape_articles()
    """
    
    def __init__(self, operation: str, **context):
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
        logger.info(f"Starting {self.operation} ({context_str})")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            logger.success(f"{self.operation} completed in {elapsed:.2f}s")
        else:
            logger.error(
                f"{self.operation} failed after {elapsed:.2f}s: {exc_val}"
            )
        
        return False  # Don't suppress exceptions


def log_dataframe_info(df, name: str = "DataFrame"):
    """
    Log pandas DataFrame information.
    
    Args:
        df: Pandas DataFrame
        name: Name for logging
    """
    logger.info(f"{name} shape: {df.shape}")
    logger.debug(f"{name} columns: {list(df.columns)}")
    logger.debug(f"{name} memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def log_model_metrics(metrics: dict, model_name: str):
    """
    Log model evaluation metrics in structured format.
    
    Args:
        metrics: Dictionary of metric names and values
        model_name: Name of the model
    """
    logger.info(f"Metrics for {model_name}:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")


# Initialize default logger on import
setup_logger()