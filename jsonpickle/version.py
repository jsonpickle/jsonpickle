try:
    import importlib_metadata as metadata
except ImportError:
    metadata = None


def _get_version():
    try:
        version = metadata.version('jsonpickle')
    except (AttributeError, ImportError):
        version = __default_version__
    return version


__default_version__ = '0.0.0-alpha'
__version__ = _get_version()
