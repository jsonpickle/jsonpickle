
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
        cls._registry.update((type_, cls) for type_ in types_handled)

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
