"""YAML extension for jsonpickle

The YAML extension module connects jsonpickle to the PyYAML `yaml` module.
"""

from typing import Optional

from ..backend import JSONBackend
from ..backend import json as jsonpickle_backend


def register(backend: Optional[JSONBackend] = None) -> bool:
    """Register the yaml module with jsonpickle's JSONBackend"""
    if backend is None:
        backend = jsonpickle_backend
    return backend.load_backend(
        "yaml", dumps="dump", loads="safe_load", loads_exc="YAMLError"
    )
