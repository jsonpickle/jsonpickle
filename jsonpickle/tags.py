"""The jsonpickle.tags module provides the custom tags
used for pickling and unpickling Python objects.

These tags are keys into the flattened dictionaries
created by the Pickler class.  The Unpickler uses
these custom key names to identify dictionaries
that need to be specially handled.
"""

BYTES: str = "py/bytes"
B64: str = "py/b64"
B85: str = "py/b85"
FUNCTION: str = "py/function"
ID: str = "py/id"
INITARGS: str = "py/initargs"
ITERATOR: str = "py/iterator"
JSON_KEY: str = "json://"
MODULE: str = "py/mod"
NEWARGS: str = "py/newargs"
NEWARGSEX: str = "py/newargsex"
NEWOBJ: str = "py/newobj"
OBJECT: str = "py/object"
PROPERTY: str = "py/property"
REDUCE: str = "py/reduce"
REF: str = "py/ref"
REPR: str = "py/repr"
SEQ: str = "py/seq"
SET: str = "py/set"
STATE: str = "py/state"
TUPLE: str = "py/tuple"
TYPE: str = "py/type"

# All reserved tag names
RESERVED: set[str] = {
    BYTES,
    FUNCTION,
    ID,
    INITARGS,
    ITERATOR,
    MODULE,
    NEWARGS,
    NEWARGSEX,
    NEWOBJ,
    OBJECT,
    PROPERTY,
    REDUCE,
    REF,
    REPR,
    SEQ,
    SET,
    STATE,
    TUPLE,
    TYPE,
}
