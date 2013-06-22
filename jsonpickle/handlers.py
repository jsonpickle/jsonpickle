"""
Custom handlers may be created to handle other objects. Each custom handler
must derive from BaseHandler and override ``.flatten`` and
``.restore``.

The handler may also declare a ``_handles`` class property which
should be a sequence of types handled by that handler. See the `mod:_handlers`
module for more examples of internal handlers implemented in jsonpickle.

A handler may also be late-bound to other types by calling the ``.handles``
method on the class. For example, the
``class:SimpleReduceHandler`` is suitable for handling objects that implement
the reduce protocol::

    @SimpleReduceHandler.handles
    class MyCustomObject(object):
        ...

        def __reduce__(self):
            return MyCustomObject, self._get_args()
"""

class TypeRegistered(type):
    """
    As classes of this metaclass are created, they keep a registry in the
    base class of all handler referenced by the keys in cls._handles.
    """
    def __init__(cls, name, bases, namespace):
        super(TypeRegistered, cls).__init__(name, bases, namespace)
        if not hasattr(cls, '_registry'):
            cls._registry = {}
        types_handled = getattr(cls, '_handles', [])
        for handled_type in types_handled:
            cls.handles(handled_type)

    def handles(handler, cls):
        """
        Register this handler for the given class
        """
        handler._registry[cls] = handler
        return cls

class BaseHandler(object):
    """
    Abstract base class for handlers.
    """

    __metaclass__ = TypeRegistered

    def __init__(self, base):
        """
        Initialize a new handler to handle `type`.

        :Parameters:
          - `base`: reference to pickler/unpickler

        """
        self._base = base

    def flatten(self, obj, data):
        """
        Flatten `obj` into a json-friendly form.

        :Parameters:
          - `obj`: object of `type`

        """
        raise NotImplementedError("Abstract method.")

    def restore(self, obj):
        """
        Restores the `obj` to `type`

        :Parameters:
          - `object`: json-friendly object

        """
        raise NotImplementedError("Abstract method.")

# for backward compatibility, provide 'registry'
# jsonpickle 0.4 clients will call it with something like:
# registry.register(handled_type, handler_class)
class registry:
    @staticmethod
    def register(handled_type, handler_class):
        handler_class.handles(handled_type)
