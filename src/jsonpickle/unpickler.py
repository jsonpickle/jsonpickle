# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett (john -at- 7oars.com)
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

import sys
import util

class Unpickler(object):
    def __init__(self):
        pass
    
    def restore(self, obj):
        """Restores a flattened object to its original python state.
        
        Simply returns any of the basic builtin types
        >>> u = Unpickler()
        >>> u.restore('hello world')
        'hello world'
        >>> u.restore({'key': 'value'})
        {'key': 'value'}
        """
        if self._isclassdict(obj):
            cls = self._loadclass(obj['classmodule__'], obj['classname__'])
            
            try:
                instance = object.__new__(cls)
            except TypeError:
                # old-style classes
                instance = cls()
            
            for k, v in obj.iteritems():
                # ignore the fake attribute
                if k in ['classmodule__', 'classname__']:
                    continue
                if k == 'classdictitems__':
                    for dictk, dictv in v.iteritems():
                        instance[dictk] = self.restore(dictv)
                    continue
                value = self.restore(v)
                if util.is_noncomplex(instance):
                    instance[k] = value
                else:
                    instance.__dict__[k] = value
            return instance
        elif util.iscollection(obj):
            # currently restores all collections to lists, even sets and tuples
            data = []
            for v in obj:
                data.append(self.restore(v))
            return data
        elif util.isdictionary(obj):
            data = {}
            for k, v in obj.iteritems():
                data[k] = self.restore(v)
            return data
        else:
            return obj
        
    def _loadclass(self, module, name):
        """Loads the module and returns the class.
        
        >>> u = Unpickler()
        >>> u._loadclass('jsonpickle.tests.classes','Thing')
        <class 'jsonpickle.tests.classes.Thing'>
        """
        __import__(module)
        mod = sys.modules[module]
        cls = getattr(mod, name)
        return cls
    
    def _isclassdict(self, obj):
        """Helper class that tests to see if the obj is a flattened object
        
        >>> u = Unpickler()
        >>> u._isclassdict({'classmodule__':'__builtin__', 'classname__':'int'})
        True
        >>> u._isclassdict({'key':'value'})    
        False
        >>> u._isclassdict(25)    
        False
        """
        if type(obj) is dict and obj.has_key('classmodule__') and obj.has_key('classname__'):
            return True
        return False
