"""
Structured logging system for DualAnimate experiments.

Provides file + console handlers with configurable formatting,
structured JSON log output for machine parsing, and integration
with the experiment tracker.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ColoredConsoleFormatter(logging.Formatter):
    """Console formatter with ANSI color codes for log levels."""

    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[41m",  # red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname:8s}{self.RESET}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON log formatter for machine-parseable structured logging.

    Each log entry is a single JSON line containing timestamp, level,
    logger name, message, and any extra fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Include exception info if present
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include any extra fields
        for key in ("step", "epoch", "metric", "value", "experiment"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry, default=str)


def setup_logging(
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
    name: str = "dualmation",
    console: bool = True,
    file: bool = True,
    json_file: bool = True,
) -> logging.Logger:
    """Configure structured logging for an experiment run.

    Sets up three handlers:
    1. Console (colored, human-readable)
    2. Text file (detailed, with timestamps)
    3. JSON file (structured, machine-parseable)

    Args:
        log_dir: Directory for log files. If None, only console logging.
        level: Logging level.
        name: Logger name.
        console: Enable console handler.
        file: Enable text file handler.
        json_file: Enable JSON file handler.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Console handler (colored)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_fmt = ColoredConsoleFormatter(
            fmt="%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler.setFormatter(console_fmt)
        logger.addHandler(console_handler)

    # File handlers
    if log_dir and (file or json_file):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if file:
            file_handler = logging.FileHandler(log_dir / "experiment.log")
            file_handler.setLevel(logging.DEBUG)  # Capture everything
            file_fmt = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_fmt)
            logger.addHandler(file_handler)

        if json_file:
            json_handler = logging.FileHandler(log_dir / "experiment.jsonl")
            json_handler.setLevel(logging.DEBUG)
            json_handler.setFormatter(JSONFormatter())
            logger.addHandler(json_handler)

    return logger
