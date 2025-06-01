"""
Stores custom jsonpickle errors.
"""

from typing import Any, Dict


class ClassNotFoundError(BaseException):
    def __init__(*args, **kwargs: Dict[str, Any]):
        pass
