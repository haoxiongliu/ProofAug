"""
Logger configuration for the Prover module.
This logger outputs to both the terminal and a log file.
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Format for log messages
LOG_FORMAT = "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO

# Log directory and file
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "prover.log"

# Ensure log directory exists
LOG_DIR.mkdir(exist_ok=True, parents=True)

# Global logger instance
_prover_logger = None


class NewLineFormatter(logging.Formatter):
    """Formatter that maintains prefix alignment for multi-line messages."""
    
    def format(self, record):
        message = logging.Formatter.format(self, record)
        if record.message != "":
            parts = message.split(record.message)
            if len(parts) > 1:
                message = message.replace("\n", "\n" + parts[0])
        return message


def setup_logger():
    """Set up the global prover logger."""
    global _prover_logger
    
    if _prover_logger is not None:
        return _prover_logger
    
    # Create logger
    logger = logging.getLogger("prover")
    logger.setLevel(DEFAULT_LOG_LEVEL)
    logger.propagate = False  # Don't propagate to parent loggers
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatters
    formatter = NewLineFormatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(DEFAULT_LOG_LEVEL)
    logger.addHandler(console_handler)
    
    # File handler with rotation (10 MB max, keep 5 backup files)
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(DEFAULT_LOG_LEVEL)
    logger.addHandler(file_handler)
    
    _prover_logger = logger
    
    # Log initialization
    logger.info(f"Prover logger initialized. Log file: {LOG_FILE}")
    
    return logger


def get_logger():
    """Get the global prover logger instance."""
    global _prover_logger
    if _prover_logger is None:
        _prover_logger = setup_logger()
    return _prover_logger


def set_log_level(level):
    """
    Set the log level for both file and console handlers.
    
    Args:
        level: Can be a string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
              or a logging level constant
    """
    logger = get_logger()
    
    # Convert string to level if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
    
    # Map level integers to names instead of using deprecated getLevelName
    level_names = {
        logging.DEBUG: 'DEBUG',
        logging.INFO: 'INFO', 
        logging.WARNING: 'WARNING',
        logging.ERROR: 'ERROR',
        logging.CRITICAL: 'CRITICAL'
    }
    level_name = level_names.get(level, f"Level {level}")
    logger.info(f"Log level set to {level_name}")


# Initialize the logger when module is imported
logger = setup_logger() 