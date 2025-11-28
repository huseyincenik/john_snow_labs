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
    patient_id: Optional[str] = None,
) -> logging.Logger:
    """Setup logger for a session and stage.

    Args:
        session_id: Session ID (eğer format {session_id}__{patient_id} ise, hasta bazlı session'dır)
        stage: Stage name (örn: "stage_extractor", "stage_consolidator")
        log_dir: Log directory (opsiyonel, varsayılan: settings.log_dir)
        patient_id: Patient ID (opsiyonel, eğer verilirse hasta bazlı alt klasör oluşturulur)
    """
    log_dir = log_dir or settings.log_dir

    # patient_session_id formatından patient_id'yi çıkar (format: {session_id}__{patient_id})
    actual_session_id = session_id
    if "__" in session_id and patient_id is None:
        parts = session_id.split("__", 1)
        if len(parts) == 2:
            # Ana session_id'yi al (ilk kısım)
            actual_session_id = parts[0]
            patient_id = parts[1]

    # Ana session klasörü oluştur
    log_path = Path(log_dir) / actual_session_id
    log_path.mkdir(parents=True, exist_ok=True)

    # Eğer patient_id varsa, hasta bazlı alt klasör oluştur
    if patient_id:
        patient_log_path = log_path / str(patient_id)
        patient_log_path.mkdir(parents=True, exist_ok=True)
        log_file = patient_log_path / f"{stage}.log"
    else:
        log_file = log_path / f"{stage}.log"

    logger = logging.getLogger(f"{actual_session_id}_{stage}_{patient_id or ''}")
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
