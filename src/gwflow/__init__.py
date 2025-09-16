"""
gwflow: Normalizing flows for gravitational-wave inference
"""

import logging
from importlib.metadata import PackageNotFoundError, version

from .gwflow import GWCalFlow

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "GWCalFlow",
]
