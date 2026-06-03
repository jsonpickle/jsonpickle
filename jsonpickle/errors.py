"""
Stores custom jsonpickle errors.
"""

from typing import Any


class ClassNotFoundError(BaseException):
    def __init__(*args: Any, **kwargs: Any):
        pass
