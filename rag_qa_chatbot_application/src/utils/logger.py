"""
Logging utilities for RAG QA Chatbot Application
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from ..config import config


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console logging"""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Setup logger with file and console handlers

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    log_level = getattr(logging, (level or config.logging.log_level).upper())
    logger.setLevel(log_level)

    # Create formatters
    file_formatter = logging.Formatter(config.logging.log_format)
    console_formatter = ColoredFormatter(config.logging.log_format)

    # File handler with rotation
    if log_to_file:
        log_file = Path(config.logging.log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=config.logging.max_bytes,
            backupCount=config.logging.backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)

    return logger


def log_function_call(logger: logging.Logger):
    """Decorator to log function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling function: {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.debug(
                    f"Function {func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(
                    f"Function {func.__name__} failed with error: {str(e)}")
                raise
        return wrapper
    return decorator


def log_performance(logger: logging.Logger):
    """Decorator to log function performance"""
    import time

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.debug(f"Starting {func.__name__}")

            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                logger.info(
                    f"{func.__name__} completed in {duration:.2f} seconds")
                return result
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                logger.error(
                    f"{func.__name__} failed after {duration:.2f} seconds: {str(e)}")
                raise

        return wrapper
    return decorator


# Create main application logger
app_logger = setup_logger("rag_chatbot")

# Export commonly used loggers
__all__ = [
    "setup_logger",
    "log_function_call",
    "log_performance",
    "app_logger",
    "ColoredFormatter"
]
