# backend/monitoring/logger.py

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from backend.config import Config

# Create logs directory
LOGS_DIR = Path(Config.DATA_DIR) / "logs"
LOGS_DIR.mkdir(exist_ok=True, parents=True)


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""

    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


def setup_logger(name: str = "pinelab", level: str = None) -> logging.Logger:
    """
    Setup application logger with console and file handlers.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    if level is None:
        level = Config.LOG_LEVEL

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        '[%(levelname)s] %(asctime)s - %(name)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)

    # File handler - rotating by size
    file_handler = RotatingFileHandler(
        LOGS_DIR / f"{name}.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Error file handler - separate file for errors
    error_handler = RotatingFileHandler(
        LOGS_DIR / f"{name}_errors.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)

    return logger


# Global logger instances
app_logger = setup_logger("pinelab")
auth_logger = setup_logger("pinelab.auth")
trade_logger = setup_logger("pinelab.trades")
optimization_logger = setup_logger("pinelab.optimization")
api_logger = setup_logger("pinelab.api")


# Convenience functions
def log_info(message: str, logger_name: str = "pinelab"):
    """Log info message."""
    logging.getLogger(logger_name).info(message)


def log_warning(message: str, logger_name: str = "pinelab"):
    """Log warning message."""
    logging.getLogger(logger_name).warning(message)


def log_error(message: str, logger_name: str = "pinelab", exc_info: bool = False):
    """Log error message."""
    logging.getLogger(logger_name).error(message, exc_info=exc_info)


def log_debug(message: str, logger_name: str = "pinelab"):
    """Log debug message."""
    logging.getLogger(logger_name).debug(message)


def log_trade(order_id: int, symbol: str, side: str, status: str, details: str = ""):
    """Log trade execution."""
    trade_logger.info(f"Order #{order_id} | {symbol} | {side.upper()} | {status} | {details}")


def log_optimization(strategy: str, trial: int, metric: float, params: dict):
    """Log optimization trial."""
    optimization_logger.info(f"{strategy} | Trial {trial} | Metric: {metric:.4f} | Params: {params}")


def log_api_request(endpoint: str, method: str, user_id: int = None, status: int = 200, duration_ms: float = 0):
    """Log API request."""
    user_str = f"User {user_id}" if user_id else "Anonymous"
    api_logger.info(f"{method} {endpoint} | {user_str} | {status} | {duration_ms:.2f}ms")
