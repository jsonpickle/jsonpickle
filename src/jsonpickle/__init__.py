# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett (john -at- 7oars.com)
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

"""Python library for serializing any arbitrary object graph into JSON

>>> import jsonpickle
>>> from jsonpickle.tests.classes import Thing

Create an object.
>>> obj = Thing('A String')
>>> print obj.name
A String

Use jsonpickle to transform the object into a JSON string.
>>> pickled = jsonpickle.encode(obj)
>>> print pickled
{"classname__": "Thing", "child": null, "name": "A String", "classmodule__": "jsonpickle.tests.classes"}

Use jsonpickle to recreate a Python object from a JSON string
>>> unpickled = jsonpickle.decode(pickled)
>>> print unpickled.name
A String

The new object has the same type and data, but essentially is now a copy of the original.
>>> obj == unpickled
False
>>> obj.name == unpickled.name
True
>>> type(obj) == type(unpickled)
True

If you will never need to load (regenerate the Python class from JSON), you can
pass in the keyword unpicklable=False to prevent extra information from being 
added to JSON.
>>> oneway = jsonpickle.encode(obj, unpicklable=False)
>>> print oneway
{"name": "A String", "child": null}

"""
    
from pickler import Pickler
from unpickler import Unpickler

__version__ = '0.1.0'
__all__ = [
    'encode', 'decode'
]

class Struct(object): pass
json = Struct()

def _use_cjson():
    import cjson
    json.encode = cjson.encode
    json.decode = cjson.decode

def _use_simplejson():
    import simplejson
    json.encode = simplejson.dumps
    json.decode = simplejson.loads

try:
    _use_cjson()
except ImportError:
    _use_simplejson()
    
def encode(value, **kwargs):
    """Returns a JSON formatted representation of value, a Python object.
    
    Optionally takes a keyword argument unpicklable.  If set to False,
    the output does not contain the information necessary to turn 
    the json back into Python.
    
    >>> encode('my string')
    '"my string"'
    >>> encode(36)
    '36'
    """
    j = Pickler(unpicklable=__isunpicklable(kwargs))
    return json.encode(j.flatten(value))

def decode(string):
    """Converts the JSON string into a Python object.
    
    >>> str(decode('"my string"'))
    'my string'
    >>> decode('36')
    36
    """
    j = Unpickler()
    return j.restore(json.decode(string))

def __isunpicklable(kw):
    """Utility function for finding keyword unpicklable and returning value.
    Default is assumed to be True.
    
    >>> __isunpicklable({})
    True
    >>> __isunpicklable({'unpicklable':True})
    True
    >>> __isunpicklable({'unpicklable':False})
    False
    
    """
    if 'unpicklable' in kw and not kw['unpicklable']:
        return False
    return True

