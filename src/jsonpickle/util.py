# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett (john -at- 7oars.com)
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

"""Helper functions for pickling and unpickling.  Most functions assist in 
determining the type of an object.
"""
import time
import datetime

COLLECTIONS = set, list, tuple
PRIMITIVES = str, unicode, int, float, bool, long
NEEDS_REPR = (datetime.datetime, datetime.time, datetime.date, 
              datetime.timedelta)

def is_primitive(obj):
    """Helper method to see if the object is a basic data type. Strings, 
    integers, longs, floats, booleans, and None are considered primitive 
    and will return True when passed into *is_primitive()*
    
    >>> is_primitive(3)
    True
    >>> is_primitive([4,4])
    False
    """
    if obj is None:
        return True
    elif type(obj) in PRIMITIVES:
        return True
    return False

def is_dictionary(obj):
    """Helper method for testing if the object is a dictionary.
    
    >>> is_dictionary({'key':'value'})
    True
    """   
    return type(obj) is dict

def is_collection(obj):
    """Helper method to see if the object is a Python collection (list, 
    set, or tuple).
    
    >>> is_collection([4])
    True
    """
    return type(obj) in COLLECTIONS

def is_dictionary_subclass(obj):
    """Returns True if *obj* is a subclass of the dict type. *obj* must be 
    a subclass and not the actual builtin dict.
    
    >>> class Temp(dict): pass
    >>> is_dictionary_subclass(Temp())
    True
    """
    return issubclass(obj.__class__, dict) and not is_dictionary(obj)

def is_collection_subclass(obj):
    """Returns True if *obj* is a subclass of a collection type, such as list
    set, tuple, etc.. *obj* must be a subclass and not the actual builtin, such
    as list, set, tuple, etc..
    
    >>> class Temp(list): pass
    >>> is_collection_subclass(Temp())
    True
    """
    #TODO add UserDict
    return issubclass(obj.__class__, COLLECTIONS) and not is_collection(obj)

def is_noncomplex(obj):
    """Returns True if *obj* is a special (weird) class, that is complex than 
    primitive data types, but is not a full object. Including:
    
        * :class:`~time.struct_time`
    """
    if type(obj) is time.struct_time:
        return True
    return False

def is_repr(obj):
    """Returns True if the *obj* must be encoded and decoded using the 
    :func:`repr` function. Including:
        
        * :class:`~datetime.datetime`
        * :class:`~datetime.date`
        * :class:`~datetime.time`
        * :class:`~datetime.timedelta`
    """
    return isinstance(obj, NEEDS_REPR)
