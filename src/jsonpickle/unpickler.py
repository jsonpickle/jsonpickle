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
    def restore(self, obj):
        """Restores a flattened object to its original python state.
        
        Simply returns any of the basic builtin types
        
        >>> u = Unpickler()
        >>> u.restore('hello world')
        'hello world'
        >>> u.restore({'key': 'value'})
        {'key': 'value'}
        """
        if isclassdict(obj):
            if 'classrepr__' in obj:
                return loadrepr(obj['classmodule__'], obj['classrepr__'])
            
            cls = loadclass(obj['classmodule__'], obj['classname__'])
            try:
                instance = object.__new__(cls)
            except TypeError:
                # old-style classes
                instance = cls()
            
            for k, v in obj.iteritems():
                # ignore the fake attribute
                if k in ('classmodule__', 'classname__'):
                    continue
                value = self.restore(v)
                if (util.is_noncomplex(instance) or
                        util.is_dictionary_subclass(instance)):
                    instance[k] = value
                else:
                    instance.__dict__[k] = value
            return instance
        elif util.is_collection(obj):
            # currently restores all collections to lists, even sets and tuples
            data = []
            for v in obj:
                data.append(self.restore(v))
            return data
        elif util.is_dictionary(obj):
            data = {}
            for k, v in obj.iteritems():
                data[k] = self.restore(v)
            return data
        else:
            return obj
        
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
