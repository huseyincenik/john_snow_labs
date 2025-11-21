"""Logging utilities."""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
from config.settings import settings


def setup_logger(
    session_id: str,
    stage: str,
    log_dir: Optional[str] = None,
) -> logging.Logger:
    """Setup logger for a session and stage."""
    log_dir = log_dir or settings.log_dir
    log_path = Path(log_dir) / session_id
    log_path.mkdir(parents=True, exist_ok=True)
    
    log_file = log_path / f"{stage}.log"
    
    logger = logging.getLogger(f"{session_id}_{stage}")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger by name."""
    return logging.getLogger(name)

