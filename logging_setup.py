"""Logging configuration helpers used across the project."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from typing import Iterable, Iterator, Sequence


class TelemetryFilter(logging.Filter):
    """Removes noisy telemetry messages from console output."""

    def __init__(self, patterns: Sequence[str] | None = None) -> None:
        super().__init__()
        self._patterns: Sequence[str] = patterns or (
            "Failed to send telemetry event",
            "telemetry",
            "capture() takes",
        )

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - простая проверка
        message = ""
        try:
            message = record.getMessage()
        except Exception:
            return True
        for pattern in self._patterns:
            if pattern in message:
                return False
        return True


class _StderrFilterWriter:
    """Wraps stderr and hides lines that contain blocked patterns."""

    def __init__(self, original, patterns: Sequence[str]) -> None:
        self._original = original
        self._patterns: Sequence[str] = patterns

    def write(self, data: str) -> None:
        if not data:
            return
        for pattern in self._patterns:
            if pattern in data:
                return
        try:
            self._original.write(data)
        except Exception:
            pass

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:
            pass


@contextmanager
def suppress_stderr_patterns(patterns: Iterable[str]) -> Iterator[None]:
    """Temporarily suppresses stderr lines that contain specific patterns."""

    original = sys.stderr
    try:
        sys.stderr = _StderrFilterWriter(original, tuple(patterns))  # type: ignore[assignment]
        yield
    finally:
        sys.stderr = original


def setup_logging(is_web: bool, log_file: str, file_format: str, console_level: str) -> logging.Logger:
    """Configures root logger with file and console handlers."""

    root_logger = logging.getLogger()

    # Удаляем старые обработчики, чтобы не дублировать вывод при повторных вызовах.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    root_logger.setLevel(logging.DEBUG)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    formatter = logging.Formatter(file_format)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    resolved_console_level = getattr(logging, console_level.upper(), logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(resolved_console_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(TelemetryFilter())
    root_logger.addHandler(console_handler)

    try:
        chroma_logger = logging.getLogger("chromadb")
        chroma_logger.disabled = True
    except Exception:
        pass

    root_logger.debug("Logging configured: file=%s, console_level=%s, is_web=%s", log_file, console_level, is_web)
    return root_logger


__all__ = [
    "setup_logging",
    "suppress_stderr_patterns",
]
