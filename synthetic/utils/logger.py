"""
Logging utilities
"""
import logging

def setup_logger(name="synthetic", level=logging.INFO):
    """Simple project-wide logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # prevent duplicate handlers if called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                      datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger