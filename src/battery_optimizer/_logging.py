"""Logging helpers shared by the CLI and the resident service."""

from __future__ import annotations

import logging

_HTTP_LOGGERS = ("httpx", "httpcore")


def quiet_http_loggers(*, verbose: bool = False) -> None:
    """Keep httpx/httpcore per-request lines out of the log unless *verbose*.

    httpx logs every request at INFO (with the full URL); a bootstrap makes ~1000 of them, which
    buries the pipeline's own progress lines.  With ``verbose`` the loggers are left alone so they
    follow the root level.
    """
    if verbose:
        return
    for name in _HTTP_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
