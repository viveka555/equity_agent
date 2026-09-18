"""
logging_config.py

Central logging configuration for the Institutional Equity Research Agent.

Responsibilities:
----------------
- Configure application-wide logging.
- Provide consistent log formatting.
- Avoid duplicated logging setup across modules.

Used by:
--------
All modules in the project.

Example:
-------
from config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
"""
from __future__ import annotations
import logging

def setup_logging(level: int=logging.INFO) ->None:
    """
    Configure the root logger for the application.

    Args:
        level: Logging level (default: INFO).

    Returns:
        None
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H-%M-%S"
    )