try:
    from importlib import metadata
except (ImportError, OSError):
    metadata = None  # type: ignore


def _get_version() -> str:
    default_version = "0.0.0-alpha"
    try:
        version = metadata.version("jsonpickle")
    except (AttributeError, ImportError, OSError):
        version = default_version
    return version


__version__ = _get_version()
