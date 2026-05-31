"""
Structured logging with latency tracking.

Uses structlog for JSON-formatted logs in production,
colored console output in DEBUG mode.
"""

import logging
import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure structlog for the application.

    Call once at startup before any logger is used.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Choose renderer based on level
    if level <= logging.DEBUG:
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Suppress noisy third-party loggers
    for name in ("urllib3", "asyncio", "SmartApi", "logzero", "websockets"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a named logger instance."""
    return structlog.get_logger(name)
