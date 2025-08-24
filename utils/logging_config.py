"""
QuantoniumOS Logging Configuration

Centralized logging configuration for all QuantoniumOS components.
Replaces scattered print() statements with proper structured logging.

Usage:
    from utils.logging_config import get_logger
    
    logger = get_logger(__name__)
    logger.info("Information message")
    logger.warning("Warning message")
    logger.error("Error message")
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


class QuantoniumFormatter(logging.Formatter):
    """Custom formatter for QuantoniumOS logs"""

    def __init__(self):
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        # Color codes for different log levels
        colors = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
        }
        reset = "\033[0m"

        # Get timestamp
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")

        # Get color for level
        color = colors.get(record.levelname, "")

        # Format message
        formatted = f"{color}[{timestamp}] {record.levelname:8} | {record.name:20} | {record.getMessage()}{reset}"

        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


def setup_logging(
    level: str = "INFO", log_file: Optional[str] = None, console: bool = True
) -> None:
    """
    Setup centralized logging for QuantoniumOS

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        console: Whether to log to console
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set root level
    root_logger.setLevel(numeric_level)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(QuantoniumFormatter())
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(numeric_level)

        # Use simple format for file logs
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Initialize default logging
setup_logging(level="INFO", log_file="quantoniumos.log", console=True)
