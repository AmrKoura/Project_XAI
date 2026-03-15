"""
Project-wide logging configuration.

Uses loguru for structured, colour-coded logging with file rotation.
"""

from loguru import logger
import sys


def setup_logger(
    log_file: str = "logs/app.log",
    level: str = "INFO",
    rotation: str = "10 MB",
) -> None:
    """Configure the project logger.

    Parameters
    ----------
    log_file : str
        Path to the log file.
    level : str
        Minimum log level.
    rotation : str
        Log file rotation size.
    """
    ...


def get_logger(name: str = "xai_replenishment"):
    """Return a named logger instance.

    Parameters
    ----------
    name : str

    Returns
    -------
    loguru.Logger
    """
    ...
