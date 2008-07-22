# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett
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
>>> pickled = jsonpickle.dumps(obj)
>>> print pickled
{"classname__": "Thing", "child": null, "name": "A String", "classmodule__": "jsonpickle.tests.classes"}

Use jsonpickle to recreate a Python object from a JSON string
>>> unpickled = jsonpickle.loads(pickled)
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
>>> oneway = jsonpickle.dumps(obj, unpicklable=False)
>>> print oneway
{"name": "A String", "child": null}

"""


__version__ = '0.0.5'
__all__ = [
    'dump', 'dumps', 'load', 'loads'
]

import simplejson as json

from pickler import Pickler
from unpickler import Unpickler


def dump(value, file, **kwargs):
    """Saves a JSON formatted representation of value into file.
    """
    j = Pickler(unpicklable=__isunpicklable(kwargs))
    json.dump(j.flatten(value), file)

def dumps(value, **kwargs):
    """Returns a JSON formatted representation of value, a Python object.
    
    Optionally takes a keyword argument unpicklable.  If set to False,
    the output does not contain the information necessary to 
    
    >>> dumps('my string')
    '"my string"'
    >>> dumps(36)
    '36'
    """
    j = Pickler(unpicklable=__isunpicklable(kwargs))
    return json.dumps(j.flatten(value))

def load(file):
    """Converts the JSON string in file into a Python object
    """
    j = Unpickler()
    return j.restore(json.load(file))

def loads(string):
    """Converts the JSON string into a Python object.
    
    >>> loads('"my string"')
    u'my string'
    >>> loads('36')
    36
    """
    j = Unpickler()
    return j.restore(json.loads(string))

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
    if kw.has_key('unpicklable') and not kw['unpicklable']:
        return False
    return True

