import gmpy2 as gmpy

from ..handlers import BaseHandler, register, unregister

__all__ = ['register_handlers', 'unregister_handlers']


class GmpyMPZHandler(BaseHandler):
    def flatten(self, obj, data):
        data['int'] = int(obj)
        return data

    def restore(self, data):
        return gmpy.mpz(data['int'])


def register_handlers():
    register(gmpy.mpz, GmpyMPZHandler, base=True)


def unregister_handlers():
    unregister(gmpy.mpz)
