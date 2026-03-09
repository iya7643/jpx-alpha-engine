from __future__ import annotations

import logging

import qlib
from qlib.constant import REG_US


LOGGER = logging.getLogger(__name__)


def _is_qlib_initialized() -> bool:
    """Return True only when qlib project path is already available."""
    try:
        return qlib.get_project_path() is not None
    except FileNotFoundError:
        return False


def init_qlib(provider_uri: str | None) -> None:
    """Initialize qlib once for the process."""
    if _is_qlib_initialized():
        return

    if provider_uri:
        LOGGER.info('Initializing qlib with provider_uri=%s', provider_uri)
        qlib.init(provider_uri=provider_uri, region=REG_US)
        return

    LOGGER.info('Initializing qlib with default provider path')
    qlib.init(region=REG_US)
