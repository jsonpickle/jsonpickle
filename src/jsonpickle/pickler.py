# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett (john -at- 7oars.com)
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

import util

class Pickler(object):
    """Converts a Python object to a JSON representation.
    
    Setting unpicklable to False removes the ability to regenerate
    the objects into object types beyond what the standard simplejson
    library supports.

    >>> p = Pickler()
    >>> p.flatten('hello world')
    'hello world'
    """
    
    def __init__(self, unpicklable=True):
        self.unpicklable = unpicklable
    
    def flatten(self, obj):
        """Takes an object and returns a JSON-safe representation of it.
        
        Simply returns any of the basic builtin datatypes
        >>> p = Pickler()
        >>> p.flatten('hello world')
        'hello world'
        >>> p.flatten(u'hello world')
        u'hello world'
        >>> p.flatten(49)
        49
        >>> p.flatten(350.0)
        350.0
        >>> p.flatten(True)
        True
        >>> p.flatten(False)
        False
        >>> r = p.flatten(None)
        >>> r is None
        True
        >>> p.flatten(False)
        False
        >>> p.flatten([1, 2, 3, 4])
        [1, 2, 3, 4]
        >>> p.flatten((1,))
        (1,)
        >>> p.flatten({'key': 'value'})
        {'key': 'value'}
        """
        
        if util.isprimitive(obj):
            return obj
        elif util.iscollection(obj):
            data = [] # obj.__class__()
            for v in obj:
                data.append(self.flatten(v))
            return obj.__class__(data)
            #TODO handle tuple and sets
        elif util.isdictionary(obj):
            data = obj.__class__()
            for k, v in obj.iteritems():
                data[k] = self.flatten(v)
            return data
        elif isinstance(obj, object):
            data = {}
            module, name = self._getclassdetail(obj)
            if self.unpicklable is True:
                data['classmodule__'] = module
                data['classname__'] = name 
            if util.is_dictionary_subclass(obj):
                for k, v in obj.iteritems():
                    data[k] = self.flatten(v)
            elif util.is_noncomplex(obj):
                data = [] # obj.__class__()
                for v in obj:
                    data.append(self.flatten(v))
            else:
                for k, v in obj.__dict__.iteritems():
                    data[str(k)] = self.flatten(v)
            return data
        # else, what else? (classes, methods, functions, old style classes...)
        
    def _getclassdetail(self, obj):
        """Helper class to return the class of an object.
        
        >>> p = Pickler()
        >>> class Klass(object): pass
        >>> p._getclassdetail(Klass())
        ('jsonpickle.pickler', 'Klass')
        >>> p._getclassdetail(25)
        ('__builtin__', 'int')
        >>> p._getclassdetail(None)
        ('__builtin__', 'NoneType')
        >>> p._getclassdetail(False)
        ('__builtin__', 'bool')
        """
        cls = getattr(obj, '__class__')
        module = getattr(cls, '__module__')
        name = getattr(cls, '__name__')
        return module, name
    
