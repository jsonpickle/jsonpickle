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
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Type,
    TypeVar,
    Union,
)

from . import tags

# key
K = TypeVar("K")
# value
V = TypeVar("V")
# type
T = TypeVar("T")

_ITERATOR_TYPE: type = type(iter(""))
SEQUENCES: tuple[type] = (list, set, tuple)  # type: ignore[assignment]
SEQUENCES_SET: set[type] = {list, set, tuple}
PRIMITIVES: set[type] = {str, bool, int, float, type(None)}
FUNCTION_TYPES: set[type] = {
    types.FunctionType,
    types.MethodType,
    types.LambdaType,
    types.BuiltinFunctionType,
    types.BuiltinMethodType,
}
# Internal set for NON_REDUCIBLE_TYPES that excludes MethodType to allow method round-trip
_NON_REDUCIBLE_FUNCTION_TYPES: set[type] = FUNCTION_TYPES - {types.MethodType}
NON_REDUCIBLE_TYPES: set[type] = (
    {
        list,
        dict,
        set,
        tuple,
        object,
        bytes,
    }
    | PRIMITIVES
    | _NON_REDUCIBLE_FUNCTION_TYPES
)
NON_CLASS_TYPES: set[type] = {
    list,
    dict,
    set,
    tuple,
    bytes,
} | PRIMITIVES
_TYPES_IMPORTABLE_NAMES: dict[Union[type, Callable[..., Any]], str] = {
    getattr(types, name): f"types.{name}"
    for name in types.__all__
    if name.endswith("Type")
}


def _is_type(obj: Any) -> bool:
    """Returns True is obj is a reference to a type.

    >>> _is_type(1)
    False

    >>> _is_type(object)
    True

    >>> class Klass: pass
    >>> _is_type(Klass)
    True
    """
    # use "isinstance" and not "is" to allow for metaclasses
    return isinstance(obj, type)


def has_method(obj: Any, name: str) -> bool:
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
    base_type = obj if _is_type(obj) else obj.__class__
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
    if not hasattr(func, "__self__"):
        return False
    bound_to = getattr(func, "__self__")

    # class methods
    if isinstance(original, classmethod):
        return issubclass(base_type, bound_to)

    # bound methods
    return isinstance(obj, type(bound_to))


def _is_object(obj: Any) -> bool:
    """Returns True is obj is a reference to an object instance.

    >>> _is_object(1)
    True

    >>> _is_object(object())
    True

    >>> _is_object(lambda x: 1)
    False
    """
    return isinstance(obj, object) and not isinstance(
        obj, (type, types.FunctionType, types.BuiltinFunctionType)
    )


def _is_not_class(obj: Any) -> bool:
    """Determines if the object is not a class or a class instance.
    Used for serializing properties.
    """
    return type(obj) in NON_CLASS_TYPES


def _is_primitive(obj: Any) -> bool:
    """Helper method to see if the object is a basic data type. Unicode strings,
    integers, longs, floats, booleans, and None are considered primitive
    and will return True when passed into *_is_primitive()*

    >>> _is_primitive(3)
    True
    >>> _is_primitive([4,4])
    False
    """
    return type(obj) in PRIMITIVES


def _is_enum(obj: Any) -> bool:
    """Is the object an enum?"""
    return "enum" in sys.modules and isinstance(obj, sys.modules["enum"].Enum)


def _is_dictionary_subclass(obj: Any) -> bool:
    """Returns True if *obj* is a subclass of the dict type. *obj* must be
    a subclass and not the actual builtin dict.

    >>> class Temp(dict): pass
    >>> _is_dictionary_subclass(Temp())
    True
    """
    # TODO: add UserDict
    return (
        hasattr(obj, "__class__")
        and issubclass(obj.__class__, dict)
        and type(obj) is not dict
    )


def _is_sequence_subclass(obj: Any) -> bool:
    """Returns True if *obj* is a subclass of list, set or tuple.

    *obj* must be a subclass and not the actual builtin, such
    as list, set, tuple, etc..

    >>> class Temp(list): pass
    >>> _is_sequence_subclass(Temp())
    True
    """
    return (
        hasattr(obj, "__class__")
        and issubclass(obj.__class__, SEQUENCES)
        and type(obj) not in SEQUENCES_SET
    )


def _is_noncomplex(obj: Any) -> bool:
    """Returns True if *obj* is a special (weird) class, that is more complex
    than primitive data types, but is not a full object. Including:

        * :class:`~time.struct_time`
    """
    return type(obj) is time.struct_time


def _is_function(obj: Any) -> bool:
    """Returns true if passed a function

    >>> _is_function(lambda x: 1)
    True

    >>> _is_function(locals)
    True

    >>> def method(): pass
    >>> _is_function(method)
    True

    >>> _is_function(1)
    False
    """
    return type(obj) in FUNCTION_TYPES


def _is_module_function(obj: Any) -> bool:
    """Return True if `obj` is a module-global function

    >>> import os
    >>> _is_module_function(os.path.exists)
    True

    >>> _is_module_function(lambda: None)
    False

    """

    return (
        hasattr(obj, "__class__")
        and isinstance(obj, (types.FunctionType, types.BuiltinFunctionType))
        and hasattr(obj, "__module__")
        and hasattr(obj, "__name__")
        and obj.__name__ != "<lambda>"
    ) or _is_cython_function(obj)


def _is_picklable(name: str, value: types.FunctionType) -> bool:
    """Return True if an object can be pickled

    >>> import os
    >>> _is_picklable('os', os)
    True

    >>> def foo(): pass
    >>> _is_picklable('foo', foo)
    True

    >>> _is_picklable('foo', lambda: None)
    False

    """
    if name in tags.RESERVED:
        return False
    return _is_module_function(value) or not _is_function(value)


def _is_installed(module: str) -> bool:
    """Tests to see if ``module`` is available on the sys.path

    >>> _is_installed('sys')
    True
    >>> _is_installed('hopefullythisisnotarealmodule')
    False

    """
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _is_list_like(obj: Any) -> bool:
    return hasattr(obj, "__getitem__") and hasattr(obj, "append")


def _is_iterator(obj: Any) -> bool:
    return isinstance(obj, abc_iterator) and not isinstance(obj, io.IOBase)


def _is_collections(obj: Any) -> bool:
    try:
        return type(obj).__module__ == "collections"
    except Exception:
        return False


def _is_reducible_sequence_subclass(obj: Any) -> bool:
    return hasattr(obj, "__class__") and issubclass(obj.__class__, SEQUENCES)


def _is_reducible(obj: Any) -> bool:
    """
    Returns false if of a type which have special casing,
    and should not have their __reduce__ methods used
    """
    # defaultdicts may contain functions which we cannot serialise
    if _is_collections(obj) and not isinstance(obj, collections.defaultdict):
        return True
    if (
        type(obj) in NON_REDUCIBLE_TYPES
        or obj is object
        or _is_dictionary_subclass(obj)
        or isinstance(obj, types.ModuleType)
        or _is_reducible_sequence_subclass(obj)
        or _is_list_like(obj)
        or isinstance(getattr(obj, "__slots__", None), _ITERATOR_TYPE)
        or (_is_type(obj) and obj.__module__ == "datetime")
    ):
        return False
    return True


def _is_cython_function(obj: Any) -> bool:
    """Returns true if the object is a reference to a Cython function"""
    return (
        callable(obj)
        and hasattr(obj, "__repr__")
        and repr(obj).startswith("<cyfunction ")
    )


def _is_readonly(obj: Any, attr: str, value: Any) -> bool:
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


def in_dict(obj: Any, key: str, default: bool = False) -> bool:
    """
    Returns true if key exists in obj.__dict__; false if not in.
    If obj.__dict__ is absent, return default
    """
    return (key in obj.__dict__) if getattr(obj, "__dict__", None) else default


def in_slots(obj: Any, key: str, default: bool = False) -> bool:
    """
    Returns true if key exists in obj.__slots__; false if not in.
    If obj.__slots__ is absent, return default
    """
    return (key in obj.__slots__) if getattr(obj, "__slots__", None) else default


def has_reduce(obj: Any) -> tuple[bool, bool]:
    """
    Tests if __reduce__ or __reduce_ex__ exists in the object dict or
    in the class dicts of every class in the MRO *except object*.

    Returns a tuple of booleans (has_reduce, has_reduce_ex)
    """

    if not _is_reducible(obj) or _is_type(obj):
        return (False, False)

    # in this case, reduce works and is desired
    # notwithstanding depending on default object
    # reduce
    if _is_noncomplex(obj):
        return (False, True)

    has_reduce = False
    has_reduce_ex = False

    REDUCE = "__reduce__"
    REDUCE_EX = "__reduce_ex__"

    # For object instance
    has_reduce = in_dict(obj, REDUCE) or in_slots(obj, REDUCE)
    has_reduce_ex = in_dict(obj, REDUCE_EX) or in_slots(obj, REDUCE_EX)

    # turn to the MRO
    for base in type(obj).__mro__:
        if _is_reducible(base):
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
    lookup = dict(__builtin__="builtins", exceptions="builtins")
    return lookup.get(module, module)


def _0_9_6_compat_untranslate(module: str) -> str:
    """Provide compatibility for pickles created with jsonpickle 0.9.6 and
    earlier, remapping `exceptions` and `__builtin__` to `builtins`.
    """
    lookup = dict(__builtin__="builtins", exceptions="builtins")
    return lookup.get(module, module)


def untranslate_module_name(module: str) -> str:
    """Rename module names mention in JSON to names that we can import

    This reverses the translation applied by translate_module_name() to
    a module name available to the current version of Python.

    """
    return _0_9_6_compat_untranslate(module)


def importable_name(cls: Union[type, Callable[..., Any]]) -> str:
    """
    >>> class Example(object):
    ...     pass

    >>> ex = Example()
    >>> importable_name(ex.__class__) == 'jsonpickle.util.Example'
    True
    >>> importable_name(type(25)) == 'builtins.int'
    True
    >>> importable_name(object().__str__.__class__) == 'types.MethodWrapperType'
    True
    >>> importable_name(False.__class__) == 'builtins.bool'
    True
    >>> importable_name(AttributeError) == 'builtins.AttributeError'
    True
    >>> import argparse
    >>> importable_name(type(argparse.ArgumentParser().add_argument)) == 'types.MethodType'
    True

    """
    types_importable_name = _TYPES_IMPORTABLE_NAMES.get(cls)
    if types_importable_name is not None:
        return types_importable_name

    # Use the fully-qualified name if available (Python >= 3.3)
    name = getattr(cls, "__qualname__", cls.__name__)
    module_name: str = getattr(cls, "__module__", "") or getattr(
        type(cls), "__module__", ""
    )
    module = translate_module_name(module_name)
    if not module:
        if hasattr(cls, "__self__"):
            if hasattr(cls.__self__, "__module__"):
                module = cls.__self__.__module__
            else:
                module = cls.__self__.__class__.__module__
    return f"{module}.{name}"


def b64encode(data: bytes) -> str:
    """
    Encode binary data to ascii text in base64. Data must be bytes.
    """
    return base64.b64encode(data).decode("ascii")


def b64decode(payload: str) -> bytes:
    """
    Decode payload - must be ascii text.
    """
    try:
        return base64.b64decode(payload)
    except (TypeError, binascii.Error):
        return b""


def b85encode(data: bytes) -> str:
    """
    Encode binary data to ascii text in base85. Data must be bytes.
    """
    return base64.b85encode(data).decode("ascii")


def b85decode(payload: bytes) -> bytes:
    """
    Decode payload - must be ascii text.
    """
    try:
        return base64.b85decode(payload)
    except (TypeError, ValueError):
        return b""


def itemgetter(
    obj: Any,
    getter: Callable[[Any], Any] = operator.itemgetter(0),
) -> str:
    return str(getter(obj))


def items(
    obj: dict[Any, Any],
    exclude: Iterable[Any] = (),
) -> Iterator[tuple[Any, Any]]:
    """
    This can't be easily replaced by dict.items() because this has the exclude parameter.
    Keep it for now.
    """
    for k, v in obj.items():
        if k in exclude:
            continue
        yield k, v


def loadclass(
    module_and_name: str, classes: Optional[Dict[str, Type[Any]]] = None
) -> Optional[Any]:
    """Loads the module and returns the class.

    >>> cls = loadclass('datetime.datetime')
    >>> cls.__name__
    'datetime'

    >>> loadclass('does.not.exist')

    >>> loadclass('builtins.int')()
    0

    """
    # Check if the class exists in a caller-provided scope
    if classes:
        try:
            return classes[module_and_name]
        except KeyError:
            # maybe they didn't provide a fully qualified path
            try:
                return classes[module_and_name.rsplit(".", 1)[-1]]
            except KeyError:
                pass
    # Otherwise, load classes from globally-accessible imports
    names = module_and_name.split(".")
    # First assume that everything up to the last dot is the module name,
    # then try other splits to handle classes that are defined within
    # classes
    for up_to in range(len(names) - 1, 0, -1):
        module = untranslate_module_name(".".join(names[:up_to]))
        try:
            __import__(module)
            obj = sys.modules[module]
            for class_name in names[up_to:]:
                obj = getattr(obj, class_name)
            return obj
        except (AttributeError, ImportError, ValueError):
            continue
    # NoneType is a special case and can not be imported/created
    if module_and_name == "builtins.NoneType":
        return type(None)
    return None
