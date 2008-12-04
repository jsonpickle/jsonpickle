# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett (john -at- 7oars.com)
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

import jsonpickle.util as util

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
        ## The current recursion depth
        self._depth = 0
        ## Maps id(obj) to reference names
        self._objs = {}
        ## The namestack grows whenever we recurse into a child object
        self._namestack = []

    def _reset(self):
        self._objs = {}
        self._namestack = []

    def _push(self):
        """Steps down one level in the namespace.
        """
        self._depth += 1

    def _pop(self, value):
        """Step up one level in the namespace and return the value.
        If we're at the root, reset the pickler's state.
        """
        self._depth -= 1
        if self._depth == 0:
            self._reset()
        return value

    def _mkref(self, obj):
        objid = id(obj)
        if objid not in self._objs:
            self._objs[objid] = '/' + '/'.join(self._namestack)
            return True
        return False

    def _getref(self, obj):
        return self._objs.get(id(obj))

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

        self._push()

        if util.is_primitive(obj):
            return self._pop(obj)
        elif util.is_collection(obj):
            return self._pop(obj.__class__([ self.flatten(v) for v in obj ]))
            #TODO handle tuple and sets
        elif util.is_dictionary(obj):
            data = obj.__class__()
            for k, v in obj.iteritems():
                self._namestack.append(k)
                data[k] = self.flatten(v)
                self._namestack.pop()
            return self._pop(data)
        elif isinstance(obj, object):
            data = {}
            if self._mkref(obj):
                module, name = _getclassdetail(obj)
                if self.unpicklable is True:
                    data['classmodule__'] = module
                    data['classname__'] = name
                if util.is_dictionary_subclass(obj):
                    for k, v in obj.iteritems():
                        self._namestack.append(k)
                        data[k] = self.flatten(v)
                        self._namestack.pop()
                elif util.is_noncomplex(obj):
                    data = []
                    for v in obj:
                        data.append(self.flatten(v))
                elif util.is_repr(obj):
                    if self.unpicklable is True:
                        data['classrepr__'] = repr(obj)
                    else:
                        data = str(obj)
                else:
                    for k, v in obj.__dict__.iteritems():
                        self._namestack.append(k)
                        data[str(k)] = self.flatten(v)
                        self._namestack.pop()
            else:
                # We've seen this object before so place an object
                # reference tag in the data. This avoids infinite recursion
                # when processing cyclical objects.
                data['objref__'] = self._getref(obj)

            return self._pop(data)
        # else, what else? (classes, methods, functions, old style classes...)
        
def _getclassdetail(obj):
    """Helper class to return the class of an object.
    
    >>> class Klass(object): pass
    >>> _getclassdetail(Klass())
    ('jsonpickle.pickler', 'Klass')
    >>> _getclassdetail(25)
    ('__builtin__', 'int')
    >>> _getclassdetail(None)
    ('__builtin__', 'NoneType')
    >>> _getclassdetail(False)
    ('__builtin__', 'bool')
    """
    cls = getattr(obj, '__class__')
    module = getattr(cls, '__module__')
    name = getattr(cls, '__name__')
    return module, name
