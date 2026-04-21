from __future__ import annotations

import logging
from typing import Final

PACKAGE_LOGGER_NAME: Final[str] = "physoce_datasets"
DEFAULT_LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)s %(message)s"
DEFAULT_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
DEFAULT_HANDLER_NAME: Final[str] = "physoce_datasets.default"


def _has_default_handler(logger: logging.Logger) -> bool:
    return any(handler.get_name() == DEFAULT_HANDLER_NAME for handler in logger.handlers)


def configure_logger(level: int | None = None) -> logging.Logger:
    """Configure and return the package logger.

    Args:
        level (int | None): Optional logging level to set for the logger.
            If None, defaults to INFO if the logger's level is NOTSET, otherwise leaves the existing level unchanged.

    Returns:
        logging.Logger: The configured package logger.

    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)

    if not _has_default_handler(logger):
        handler = logging.StreamHandler()
        handler.set_name(DEFAULT_HANDLER_NAME)
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT))
        logger.addHandler(handler)

    if level is not None:
        logger.setLevel(level)
    elif logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)

    # Keep output scoped to this package logger and its children.
    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the package namespace.

    Args:
        name (str | None): Optional name for the logger.
            If None or equal to the package logger name, returns the package logger.
            If a child logger name is provided, returns that child logger.

    Returns:
        logging.Logger: The requested logger under the package namespace.

    """
    configure_logger()

    if name is None or name == PACKAGE_LOGGER_NAME:
        return logging.getLogger(PACKAGE_LOGGER_NAME)
    if name.startswith(f"{PACKAGE_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name}")


logger = get_logger()
