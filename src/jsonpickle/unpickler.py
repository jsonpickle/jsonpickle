# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett (john -at- 7oars.com)
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

import sys
import jsonpickle.util as util

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
        if isobjref(obj) and obj['objref__'] in self._namedict:
            return self._pop(self._namedict.get(obj['objref__']))

        elif isclassdict(obj):
            if 'classrepr__' in obj:
                return loadrepr(obj['classmodule__'], obj['classrepr__'])
            
            cls = loadclass(obj['classmodule__'], obj['classname__'])
            try:
                instance = object.__new__(cls)
            except TypeError:
                # old-style classes
                instance = cls()
            
            # keep a obj->name mapping for use in the _isobjref() case
            name = '/' + '/'.join(self._namestack)
            if name not in self._namedict:
                self._namedict[name] = instance

            for k, v in obj.iteritems():
                # ignore the fake attribute
                if k in ('classmodule__', 'classname__'):
                    continue
                # step into the namespace
                self._namestack.append(k)
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
            # currently restores all collections to lists, even sets and tuples
            data = []
            for v in obj:
                data.append(self.restore(v))
            return self._pop(data)
        elif util.is_dictionary(obj):
            data = {}
            for k, v in obj.iteritems():
                self._namestack.append(k)
                data[k] = self.restore(v)
                self._namestack.pop()
            return self._pop(data)
        else:
            return self._pop(obj)
        
def loadclass(module, name):
    """Loads the module and returns the class.
    
    >>> loadclass('jsonpickle.tests.classes','Thing')
    <class 'jsonpickle.tests.classes.Thing'>
    """
    __import__(module)
    mod = sys.modules[module]
    cls = getattr(mod, name)
    return cls

def loadrepr(module, objrepr):
    """Returns an instance of the object from the object's repr() string. It
    involves the dynamic specification of code.
    
    >>> loadrepr('jsonpickle.tests.classes','jsonpickle.tests.classes.Thing("json")')
    jsonpickle.tests.classes.Thing("json")
    """
    exec('import %s' % module)
    return eval(objrepr) 
    
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

def isobjref(obj):
    """Helper class that tests to see if the obj is an object reference
    
    >>> isobjref({'objref__':'/'})
    True
    >>> isobjref({'key':'value'})
    False
    >>> isobjref(25)
    False
    """
    return type(obj) is dict and 'objref__' in obj

