try:
    import gmpy2 as gmpy  # type: ignore[import-untyped]
except ImportError:
    gmpy = None

from typing import Any, Dict

from ..handlers import BaseHandler, HandlerReturn, register, unregister

__all__ = ['register_handlers', 'unregister_handlers']


class GmpyMPZHandler(BaseHandler):
    def flatten(self, obj: gmpy.mpz, data: Dict[str, Any]) -> HandlerReturn:
        data['int'] = int(obj)
        return data

    def restore(self, data: Dict[str, Any]) -> gmpy.mpz:
        return gmpy.mpz(data['int'])


def register_handlers() -> None:
    if gmpy is not None:
        register(gmpy.mpz, GmpyMPZHandler, base=True)


def unregister_handlers() -> None:
    if gmpy is not None:
        unregister(gmpy.mpz)
