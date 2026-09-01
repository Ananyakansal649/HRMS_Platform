"""
Enterprise HR AI — Application Logger
Structured logging for app lifecycle, predictions, and errors.
"""
import logging
import sys
from datetime import datetime


def get_logger(name: str = "hr_ai") -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


# Module-level loggers for common use
api_logger = get_logger("hr_ai.api")
model_logger = get_logger("hr_ai.model")
monitoring_logger = get_logger("hr_ai.monitoring")
