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

COLLECTIONS = set, list, tuple,
#TODO add PRIMITIVES global
#TODO refactor names to be consistent

def isprimitive(obj):
    """Helper method to see if the object is a basic data type. Strings, 
    integers, longs, floats, booleans, and None are considered primitive 
    and will return True when passed into *isprimitive()*
    
    >>> isprimitive(3)
    True
    >>> isprimitive([4,4])
    False
    """
    if obj is None:
        return True
    elif type(obj) in [str, unicode, int, float, bool, long]:
        return True
    return False

def isdictionary(obj):
    """Helper method for testing if the object is a dictionary.
    
    >>> isdictionary({'key':'value'})
    True
    """   
    if type(obj) is dict:
        return True
    return False

def iscollection(obj):
    """Helper method to see if the object is a Python collection (list, 
    set, or tuple).
    
    >>> iscollection([4])
    True
    """
    if type(obj) in COLLECTIONS:
        return True
    return False

def is_dictionary_subclass(obj):
    """Returns True if *obj* is a subclass of the dict type. *obj* must be 
    a subclass and not the actual builtin dict.
    
    >>> class Temp(dict): pass
    >>> is_dictionary_subclass(Temp())
    True
    """
    #TODO add UserDict
    if issubclass(obj.__class__, dict) and not isdictionary(obj):
        return True
    return False

def is_collection_subclass(obj):
    """Returns True if *obj* is a subclass of a collection type, such as list
    set, tuple, etc.. *obj* must be a subclass and not the actual builtin, such
    as list, set, tuple, etc..
    
    >>> class Temp(list): pass
    >>> is_collection_subclass(Temp())
    True
    """
    #TODO add UserDict
    if issubclass(obj.__class__, COLLECTIONS) and not iscollection(obj):
        return True
    return False

def is_noncomplex(obj):
    """Returns True if *obj* is a special (weird) class, that is complex than 
    primitive data types, but is not a full object. Including:
    
        * :class:`~time.struct_time`
    """
    if type(obj) is time.struct_time:
        return True
    return False

