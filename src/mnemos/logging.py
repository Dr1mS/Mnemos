"""Logging structuré (§6) — structlog.

JSON renderer en prod, console renderer en dev (LOG_LEVEL=DEBUG).
Anti-pattern 10 : jamais de contenu utilisateur brut hors DEBUG —
log les IDs, salience scores, timings.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    dev_mode = level <= logging.DEBUG

    logging.basicConfig(stream=sys.stderr, level=level, format="%(message)s")

    renderer: structlog.typing.Processor = (
        structlog.dev.ConsoleRenderer() if dev_mode else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
