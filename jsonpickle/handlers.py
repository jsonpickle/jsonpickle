# -*- coding: utf-8 -*-
#
class BaseHandler( object ):
    """
    Abstract base class for handlers.
    """
    def __init__(self, base):
        """
        Initializes a new handler to handle `type`.
        
        :Parameters:
          - `base`: reference to pickler/unpickler
        """
        self._base = base

    def flatten(self, object, data):
        """
        Flattens the `object` into a json-friendly form.
        
        :Parameters:
          - `object`: object of `type`
        """
        raise NotImplementedError("Abstract method.")

    def restore(self, object):
        """
        Restores the `object` to `type`
        
        :Parameters:
          - `object`: json-friendly object
        """
        raise NotImplementedError("Abstract method.")


class Registry( object ):
    REGISTRY = {}
    
    def register(self, type, handler):
        """
        Register handler.
        
        :Parameters:
          - `handler`: `BaseHandler` subclass
        """
        self.REGISTRY[type] = handler
        return handler

    def get(self, cls):
        """
        Get the customer handler for `object` (if any)
        
        :Parameters:
          - `cls`: class to handle
        """
        return self.REGISTRY.get(cls, None)

registry = Registry()
    
    
