# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett (john -at- 7oars.com)
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

import sys
import jsonpickle.util as util

## These are property names used to track object types and references
RESERVED_NAMES = (
    'classmodule__',
    'classname__',
    'classrefname__',
    'typemodule__',
    'typename__',
)

class Unpickler(object):
    def __init__(self):
        ## The current recursion depth
        self._depth = 0
        ## Maps reference names to object instances
        self._namedict = {}
        ## The namestack grows whenever we recurse into a child object
        self._namestack = []

    def _reset(self):
        """Resets the object's internal state.
        """
        self._namedict = {}
        self._namestack = []

    def _push(self):
        """Steps down one level in the namespace.
        """
        self._depth += 1

    def _pop(self, value):
        """Step up one level in the namespace and return the value.
        If we're at the root, reset the unpickler's state.
        """
        self._depth -= 1
        if self._depth == 0:
            self._reset()
        return value

    def restore(self, obj):
        """Restores a flattened object to its original python state.
        
        Simply returns any of the basic builtin types
        
        >>> u = Unpickler()
        >>> u.restore('hello world')
        'hello world'
        >>> u.restore({'key': 'value'})
        {'key': 'value'}
        """
        self._push()

        if isobjrefdict(obj):
            return self._pop(self._namedict.get(obj['objref__']))

        elif istypedict(obj):
            typeref = loadclass(obj['typemodule__'], obj['typename__'])
            if not typeref:
                return self._pop(obj)
            return self._pop(typeref)

        elif isclassdict(obj):
            if 'classrepr__' in obj:
                return self._pop(loadrepr(obj['classmodule__'],
                                          obj['classrepr__']))

            cls = loadclass(obj['classmodule__'], obj['classname__'])
            if not cls:
                return self._pop(obj)
            try:
                instance = object.__new__(cls)
            except TypeError:
                # old-style classes
                try:
                    instance = cls()
                except TypeError:
                    # fail gracefully if the constructor requires arguments
                    self._mkref(obj)
                    return self._pop(obj)
            
            # keep a obj->name mapping for use in the _isobjref() case
            self._mkref(instance)

            for k, v in obj.iteritems():
                # ignore the reserved attribute
                if k in RESERVED_NAMES:
                    continue
                self._namestack.append(k)
                # step into the namespace
                value = self.restore(v)
                if (util.is_noncomplex(instance) or
                        util.is_dictionary_subclass(instance)):
                    instance[k] = value
                else:
                    instance.__dict__[k] = value
                # step out
                self._namestack.pop()
            return self._pop(instance)

        elif util.is_collection(obj):
            return self._pop([self.restore(v) for v in obj])

        elif util.is_dictionary(obj):
            data = {}
            for k, v in obj.iteritems():
                self._namestack.append(k)
                data[k] = self.restore(v)
                self._namestack.pop()
            return self._pop(data)

        else:
            return self._pop(obj)

    def _refname(self):
        return '/' + '/'.join(self._namestack)

    def _mkref(self, obj):
        name = self._refname()
        if name not in self._namedict:
            self._namedict[name] = obj
        return name

def loadclass(module, name):
    """Loads the module and returns the class.
    
    >>> loadclass('jsonpickle.tests.classes','Thing')
    <class 'jsonpickle.tests.classes.Thing'>

    >>> loadclass('example.module.does.not.exist', 'Missing')
    

    >>> loadclass('jsonpickle.tests.classes', 'MissingThing')
    

    """
    try:
        __import__(module)
        return getattr(sys.modules[module], name)
    except:
        return None

def loadrepr(module, objrepr):
    """Returns an instance of the object from the object's repr() string. It
    involves the dynamic specification of code.
    
    >>> loadrepr('jsonpickle.tests.classes','jsonpickle.tests.classes.Thing("json")')
    jsonpickle.tests.classes.Thing("json")
    """
    exec('import %s' % module)
    return eval(objrepr) 

def istypedict(obj):
    """Helper class that tests to see if the obj is a flattened type
    
    >>> istypedict({'typemodule__': '__builtin__', 'typename__': 'object'})
    True
    >>> istypedict({'key':'value'})    
    False
    >>> istypedict(25)
    False
    """
    return type(obj) is dict and 'typemodule__' in obj and 'typename__' in obj

def isclassref(obj):
    """Helper class that tests to see if the obj is a flattened object
    
    >>> isclassref({'classmodule__':'__builtin__', 'classrefname__':'int'})
    True
    >>> isclassref({'key':'value'})    
    False
    >>> isclassdict(25)
    False
    """
    return (type(obj) is dict and
            'classmodule__' in obj and 'classrefname__' in obj)
    
def isclassdict(obj):
    """Helper class that tests to see if the obj is a flattened object
    
    >>> isclassdict({'classmodule__':'__builtin__', 'classname__':'int'})
    True
    >>> isclassdict({'key':'value'})    
    False
    >>> isclassdict(25)
    False
    """
    return type(obj) is dict and 'classmodule__' in obj and 'classname__' in obj

def isobjrefdict(obj):
    """Helper class that tests to see if the obj is an object reference
    
    >>> isobjrefdict({'objref__':'/'})
    True
    >>> isobjrefdict({'key':'value'})
    False
    >>> isobjrefdict(25)
    False
    """
    return type(obj) is dict and 'objref__' in obj

