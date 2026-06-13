"""
Enterprise RAG - Logger Utility
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from config import config


def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger instance."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, config.logging.level))
    formatter = logging.Formatter(config.logging.format)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    # Fix Unicode encoding issues on Windows console (cp1252)
    if hasattr(ch.stream, 'reconfigure'):
        try:
            ch.stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    logger.addHandler(ch)

    # File handler with rotation
    try:
        Path(config.logging.log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            config.logging.log_file,
            maxBytes=config.logging.max_bytes,
            backupCount=config.logging.backup_count
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception:
        pass

    return logger
