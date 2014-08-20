"""The jsonpickle.tags module provides the custom tags
used for pickling and unpickling Python objects.

These tags are keys into the flattened dictionaries
created by the Pickler class.  The Unpickler uses
these custom key names to identify dictionaries
that need to be specially handled.
"""
from jsonpickle.compat import set


FUNCTION = 'py/function'
ID = 'py/id'
JSON_KEY = 'json://'
NEWARGS = 'py/newargs'
OBJECT = 'py/object'
REPR = 'py/repr'
REF = 'py/ref'
STATE = 'py/state'
SET = 'py/set'
SEQ = 'py/seq'
TUPLE = 'py/tuple'
TYPE = 'py/type'

# All reserved tag names
RESERVED = set([
    FUNCTION,
    ID,
    NEWARGS,
    OBJECT,
    REF,
    REPR,
    SEQ,
    SET,
    STATE,
    TUPLE,
    TYPE,
])
