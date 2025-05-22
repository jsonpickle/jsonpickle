# Copyright (C) 2008 John Paulett (john -at- paulett.org)
# Copyright (C) 2009-2018 David Aguilar (davvid -at- gmail.com)
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

"""Helper functions for pickling and unpickling.  Most functions assist in
determining the type of an object.
"""
import base64
import binascii
import collections
import inspect
import io
import operator
import sys
import time
import types
from collections.abc import Iterator as abc_iterator
from typing import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Tuple,
    TypeVar,
)

from . import tags

# key
K = TypeVar("K")
# value
V = TypeVar("V")
# type
T = TypeVar("T")
# return value
R = TypeVar("R")

_ITERATOR_TYPE: type = type(iter(''))
SEQUENCES: tuple = (list, set, tuple)
SEQUENCES_SET: set = {list, set, tuple}
PRIMITIVES: set = {str, bool, int, float, type(None)}
FUNCTION_TYPES: set = {
    types.FunctionType,
    types.MethodType,
    types.LambdaType,
    types.BuiltinFunctionType,
    types.BuiltinMethodType,
}
NON_REDUCIBLE_TYPES: set = (
    {
        list,
        dict,
        set,
        tuple,
        object,
        bytes,
    }
    | PRIMITIVES
    | FUNCTION_TYPES
)
NON_CLASS_TYPES: set = {
    list,
    dict,
    set,
    tuple,
    bytes,
} | PRIMITIVES


def is_type(obj: object) -> bool:
    """Returns True is obj is a reference to a type.

    >>> is_type(1)
    False

    >>> is_type(object)
    True

    >>> class Klass: pass
    >>> is_type(Klass)
    True
    """
    # use "isinstance" and not "is" to allow for metaclasses
    return isinstance(obj, type)


def has_method(obj: object, name: str) -> bool:
    # false if attribute doesn't exist
    if not hasattr(obj, name):
        return False
    func = getattr(obj, name)

    # builtin descriptors like __getnewargs__
    if isinstance(func, types.BuiltinMethodType):
        return True

    # note that FunctionType has a different meaning in py2/py3
    if not isinstance(func, (types.MethodType, types.FunctionType)):
        return False

    # need to go through __dict__'s since in py3
    # methods are essentially descriptors

    # __class__ for old-style classes
    base_type = obj if is_type(obj) else obj.__class__
    original = None
    # there is no .mro() for old-style classes
    for subtype in inspect.getmro(base_type):
        original = vars(subtype).get(name)
        if original is not None:
            break

    # name not found in the mro
    if original is None:
        return False

    # static methods are always fine
    if isinstance(original, staticmethod):
        return True

    # at this point, the method has to be an instancemthod or a classmethod
    if not hasattr(func, '__self__'):
        return False
    bound_to = getattr(func, '__self__')

    # class methods
    if isinstance(original, classmethod):
        return issubclass(base_type, bound_to)

    # bound methods
    return isinstance(obj, type(bound_to))


def is_object(obj: object) -> bool:
    """Returns True is obj is a reference to an object instance.

    >>> is_object(1)
    True

    >>> is_object(object())
    True

    >>> is_object(lambda x: 1)
    False
    """
    return isinstance(obj, object) and not isinstance(
        obj, (type, types.FunctionType, types.BuiltinFunctionType)
    )


def is_not_class(obj: object) -> bool:
    """Determines if the object is not a class or a class instance.
    Used for serializing properties.
    """
    return type(obj) in NON_CLASS_TYPES


def is_primitive(obj: object) -> bool:
    """Helper method to see if the object is a basic data type. Unicode strings,
    integers, longs, floats, booleans, and None are considered primitive
    and will return True when passed into *is_primitive()*

    >>> is_primitive(3)
    True
    >>> is_primitive([4,4])
    False
    """
    return type(obj) in PRIMITIVES


def is_enum(obj: object) -> bool:
    """Is the object an enum?"""
    return 'enum' in sys.modules and isinstance(obj, sys.modules['enum'].Enum)


def is_sequence(obj: object) -> bool:
    """Helper method to see if the object is a sequence (list, set, or tuple).

    >>> is_sequence([4])
    True

    """
    return type(obj) in SEQUENCES_SET


def is_dictionary_subclass(obj: object) -> bool:
    """Returns True if *obj* is a subclass of the dict type. *obj* must be
    a subclass and not the actual builtin dict.

    >>> class Temp(dict): pass
    >>> is_dictionary_subclass(Temp())
    True
    """
    # TODO: add UserDict
    return (
        hasattr(obj, '__class__')
        and issubclass(obj.__class__, dict)
        and type(obj) is not dict
    )


def is_sequence_subclass(obj: object) -> bool:
    """Returns True if *obj* is a subclass of list, set or tuple.

    *obj* must be a subclass and not the actual builtin, such
    as list, set, tuple, etc..

    >>> class Temp(list): pass
    >>> is_sequence_subclass(Temp())
    True
    """
    return (
        hasattr(obj, '__class__')
        and issubclass(obj.__class__, SEQUENCES)
        and not is_sequence(obj)
    )


def is_noncomplex(obj: object) -> bool:
    """Returns True if *obj* is a special (weird) class, that is more complex
    than primitive data types, but is not a full object. Including:

        * :class:`~time.struct_time`
    """
    if type(obj) is time.struct_time:
        return True
    return False


def is_function(obj: object) -> bool:
    """Returns true if passed a function

    >>> is_function(lambda x: 1)
    True

    >>> is_function(locals)
    True

    >>> def method(): pass
    >>> is_function(method)
    True

    >>> is_function(1)
    False
    """
    return type(obj) in FUNCTION_TYPES


def is_module_function(obj: object) -> bool:
    """Return True if `obj` is a module-global function

    >>> import os
    >>> is_module_function(os.path.exists)
    True

    >>> is_module_function(lambda: None)
    False

    """

    return (
        hasattr(obj, '__class__')
        and isinstance(obj, (types.FunctionType, types.BuiltinFunctionType))
        and hasattr(obj, '__module__')
        and hasattr(obj, '__name__')
        and obj.__name__ != '<lambda>'
    ) or is_cython_function(obj)


def is_picklable(name: str, value: types.FunctionType) -> bool:
    """Return True if an object can be pickled

    >>> import os
    >>> is_picklable('os', os)
    True

    >>> def foo(): pass
    >>> is_picklable('foo', foo)
    True

    >>> is_picklable('foo', lambda: None)
    False

    """
    if name in tags.RESERVED:
        return False
    return is_module_function(value) or not is_function(value)


def is_installed(module: types.ModuleType) -> bool:
    """Tests to see if ``module`` is available on the sys.path

    >>> is_installed('sys')
    True
    >>> is_installed('hopefullythisisnotarealmodule')
    False

    """
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def is_list_like(obj: object) -> bool:
    return hasattr(obj, '__getitem__') and hasattr(obj, 'append')


def is_iterator(obj: object) -> bool:
    return isinstance(obj, abc_iterator) and not isinstance(obj, io.IOBase)


def is_collections(obj: object) -> bool:
    try:
        return type(obj).__module__ == 'collections'
    except Exception:
        return False


def is_reducible_sequence_subclass(obj: object) -> bool:
    return hasattr(obj, '__class__') and issubclass(obj.__class__, SEQUENCES)


def is_reducible(obj: object) -> bool:
    """
    Returns false if of a type which have special casing,
    and should not have their __reduce__ methods used
    """
    # defaultdicts may contain functions which we cannot serialise
    if is_collections(obj) and not isinstance(obj, collections.defaultdict):
        return True
    if (
        type(obj) in NON_REDUCIBLE_TYPES
        or obj is object
        or is_dictionary_subclass(obj)
        or isinstance(obj, types.ModuleType)
        or is_reducible_sequence_subclass(obj)
        or is_list_like(obj)
        or isinstance(getattr(obj, '__slots__', None), _ITERATOR_TYPE)
        or (is_type(obj) and obj.__module__ == 'datetime')
    ):
        return False
    return True


def is_cython_function(obj: object) -> bool:
    """Returns true if the object is a reference to a Cython function"""
    return (
        callable(obj)
        and hasattr(obj, '__repr__')
        and repr(obj).startswith('<cyfunction ')
    )


def is_readonly(obj: object, attr: str, value: object) -> bool:
    # CPython 3.11+ has 0-cost try/except, please use up-to-date versions!
    try:
        setattr(obj, attr, value)
        return False
    except AttributeError:
        # this is okay, it means the attribute couldn't be set
        return True
    except TypeError:
        # this should only be happening when obj is a dict
        # as these errors happen when attr isn't a str
        return True


def in_dict(obj: object, key: str, default: bool = False) -> bool:
    """
    Returns true if key exists in obj.__dict__; false if not in.
    If obj.__dict__ is absent, return default
    """
    return (key in obj.__dict__) if getattr(obj, '__dict__', None) else default


def in_slots(obj: object, key: str, default: bool = False) -> bool:
    """
    Returns true if key exists in obj.__slots__; false if not in.
    If obj.__slots__ is absent, return default
    """
    return (key in obj.__slots__) if getattr(obj, '__slots__', None) else default


def has_reduce(obj: object) -> Tuple[bool, bool]:
    """
    Tests if __reduce__ or __reduce_ex__ exists in the object dict or
    in the class dicts of every class in the MRO *except object*.

    Returns a tuple of booleans (has_reduce, has_reduce_ex)
    """

    if not is_reducible(obj) or is_type(obj):
        return (False, False)

    # in this case, reduce works and is desired
    # notwithstanding depending on default object
    # reduce
    if is_noncomplex(obj):
        return (False, True)

    has_reduce = False
    has_reduce_ex = False

    REDUCE = '__reduce__'
    REDUCE_EX = '__reduce_ex__'

    # For object instance
    has_reduce = in_dict(obj, REDUCE) or in_slots(obj, REDUCE)
    has_reduce_ex = in_dict(obj, REDUCE_EX) or in_slots(obj, REDUCE_EX)

    # turn to the MRO
    for base in type(obj).__mro__:
        if is_reducible(base):
            has_reduce = has_reduce or in_dict(base, REDUCE)
            has_reduce_ex = has_reduce_ex or in_dict(base, REDUCE_EX)
        if has_reduce and has_reduce_ex:
            return (has_reduce, has_reduce_ex)

    # for things that don't have a proper dict but can be
    # getattred (rare, but includes some builtins)
    cls = type(obj)
    object_reduce = getattr(object, REDUCE)
    object_reduce_ex = getattr(object, REDUCE_EX)
    if not has_reduce:
        has_reduce_cls = getattr(cls, REDUCE, False)
        if has_reduce_cls is not object_reduce:
            has_reduce = has_reduce_cls

    if not has_reduce_ex:
        has_reduce_ex_cls = getattr(cls, REDUCE_EX, False)
        if has_reduce_ex_cls is not object_reduce_ex:
            has_reduce_ex = has_reduce_ex_cls

    return (has_reduce, has_reduce_ex)


def translate_module_name(module: str) -> str:
    """Rename builtin modules to a consistent module name.

    Prefer the more modern naming.

    This is used so that references to Python's `builtins` module can
    be loaded in both Python 2 and 3.  We remap to the "__builtin__"
    name and unmap it when importing.

    Map the Python2 `exceptions` module to `builtins` because
    `builtins` is a superset and contains everything that is
    available in `exceptions`, which makes the translation simpler.

    See untranslate_module_name() for the reverse operation.
    """
    lookup = dict(__builtin__='builtins', exceptions='builtins')
    return lookup.get(module, module)


def _0_9_6_compat_untranslate(module: str) -> str:
    """Provide compatibility for pickles created with jsonpickle 0.9.6 and
    earlier, remapping `exceptions` and `__builtin__` to `builtins`.
    """
    lookup = dict(__builtin__='builtins', exceptions='builtins')
    return lookup.get(module, module)


def untranslate_module_name(module: str) -> str:
    """Rename module names mention in JSON to names that we can import

    This reverses the translation applied by translate_module_name() to
    a module name available to the current version of Python.

    """
    return _0_9_6_compat_untranslate(module)


def importable_name(cls: type) -> str:
    """
    >>> class Example(object):
    ...     pass

    >>> ex = Example()
    >>> importable_name(ex.__class__) == 'jsonpickle.util.Example'
    True
    >>> importable_name(type(25)) == 'builtins.int'
    True
    >>> importable_name(None.__class__) == 'builtins.NoneType'
    True
    >>> importable_name(False.__class__) == 'builtins.bool'
    True
    >>> importable_name(AttributeError) == 'builtins.AttributeError'
    True

    """
    # Use the fully-qualified name if available (Python >= 3.3)
    name = getattr(cls, '__qualname__', cls.__name__)
    module = translate_module_name(cls.__module__)
    if not module:
        if hasattr(cls, '__self__'):
            if hasattr(cls.__self__, '__module__'):
                module = cls.__self__.__module__
            else:
                module = cls.__self__.__class__.__module__
    return f'{module}.{name}'


def b64encode(data: bytes) -> str:
    """
    Encode binary data to ascii text in base64. Data must be bytes.
    """
    return base64.b64encode(data).decode('ascii')


def b64decode(payload: str) -> bytes:
    """
    Decode payload - must be ascii text.
    """
    try:
        return base64.b64decode(payload)
    except (TypeError, binascii.Error):
        return b''


def b85encode(data: bytes) -> str:
    """
    Encode binary data to ascii text in base85. Data must be bytes.
    """
    return base64.b85encode(data).decode('ascii')


def b85decode(payload: bytes) -> bytes:
    """
    Decode payload - must be ascii text.
    """
    try:
        return base64.b85decode(payload)
    except (TypeError, ValueError):
        return b''


def itemgetter(obj: T, getter: Callable[[T], R] = operator.itemgetter(0)) -> str:
    return str(getter(obj))


def items(obj: Mapping[K, V], exclude: Iterable[K] = ()) -> Iterator[Tuple[K, V]]:
    """
    This can't be easily replaced by dict.items() because this has the exclude parameter.
    Keep it for now.
    """
    for k, v in obj.items():
        if k in exclude:
            continue
        yield k, v
