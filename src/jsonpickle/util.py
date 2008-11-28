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

def isprimitive(obj):
    """Helper class to see if the object is one of the basic Python
    builtin objects.    
    
    >>> isprimitive(3)
    True
    >>> isprimitive(3.5)
    True
    >>> isprimitive(long(4))
    True
    >>> isprimitive('hello world')
    True
    >>> isprimitive(u'hello world')
    True
    >>> isprimitive(True)
    True
    >>> isprimitive(None)
    True
    >>> isprimitive([4,4])
    False
    >>> isprimitive({'key':'value'})
    False
    >>> isprimitive((1,3))
    False
    >>> isprimitive(object())
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
    >>> iscollection((4,3))
    True
    >>> iscollection(set([4,5]))
    True
    >>> iscollection({'key':'value'})
    False
    """
    
    if type(obj) in COLLECTIONS:
        return True
    return False

def is_dictionary_subclass(obj):
    """Dictionary subclass
    
    >>> class Temp(dict): pass
    >>> obj = Temp()
    >>> obj['key'] = 1
    >>> is_dictionary_subclass(Temp())
    True
    >>> is_dictionary_subclass({'a': 1})
    False
    >>> is_dictionary_subclass([1])
    False
    >>> is_dictionary_subclass('a')
    False
    """
    #TODO add UserDict
    if issubclass(obj.__class__, dict) and not isdictionary(obj):
        return True
    return False

def is_collection_subclass(obj):
    """Collection subclass
    
    >>> class Temp(list): pass
    >>> obj = Temp()
    >>> obj.append(1)
    >>> is_collection_subclass(Temp())
    True
    >>> is_collection_subclass({'a': 1})
    False
    >>> is_collection_subclass([1])
    False
    >>> is_collection_subclass('a')
    False
    """
    #TODO add UserDict
    if issubclass(obj.__class__, COLLECTIONS) and not iscollection(obj):
        return True
    return False


def is_noncomplex(obj):
    """Special (weird) classes.
    
    >>> t = time.struct_time('123456789')
    >>> print t
    ('1', '2', '3', '4', '5', '6', '7', '8', '9')
    >>> is_noncomplex(t)
    True
    >>> is_noncomplex('a')
    False
    """
    if type(obj) is time.struct_time:
        return True
    return False

