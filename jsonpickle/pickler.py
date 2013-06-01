# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett (john -at- paulett.org)
# Copyright (C) 2009, 2011, 2013 David Aguilar
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.
import operator
import jsonpickle.util as util
import jsonpickle.tags as tags
import jsonpickle.handlers as handlers
from jsonpickle.compat import unicode


class Pickler(object):
    """Converts a Python object to a JSON representation.

    Setting unpicklable to False removes the ability to regenerate
    the objects into object types beyond what the standard
    simplejson library supports.

    Setting max_depth to a negative number means there is no
    limit to the depth jsonpickle should recurse into an
    object.  Setting it to zero or higher places a hard limit
    on how deep jsonpickle recurses into objects, dictionaries, etc.

    >>> p = Pickler()
    >>> p.flatten('hello world')
    'hello world'
    """

    def __init__(self, unpicklable=True, max_depth=None):
        self.unpicklable = unpicklable
        ## The current recursion depth
        self._depth = -1
        ## The maximal recursion depth
        self._max_depth = max_depth
        ## Maps id(obj) to reference IDs
        self._objs = {}

    def _reset(self):
        self._objs = {}

    def _push(self):
        """Steps down one level in the namespace.
        """
        self._depth += 1

    def _pop(self, value):
        """Step up one level in the namespace and return the value.
        If we're at the root, reset the pickler's state.
        """
        self._depth -= 1
        if self._depth == -1:
            self._reset()
        return value

    def _mkref(self, obj):
        # Do not use references if not unpicklable.
        if self.unpicklable is False:
            return True
        objid = id(obj)
        if objid not in self._objs:
            new_id = len(self._objs)
            self._objs[objid] = new_id
            return True
        return False

    def _getref(self, obj):
        return {tags.ID: self._objs.get(id(obj))}

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
        >>> p.flatten((1,2,))[tags.TUPLE]
        [1, 2]
        >>> p.flatten({'key': 'value'})
        {'key': 'value'}
        """
        self._push()

        max_reached = self._depth == self._max_depth
        flatten_func = self._get_flattener(obj) if not max_reached else repr

        return self._pop(flatten_func(obj))

    def _get_flattener(self, obj):

        if util.is_primitive(obj):
            return lambda obj: obj

        list_recurse = lambda obj: [self.flatten(v) for v in obj]

        if util.is_list(obj):
            if self._mkref(obj):
                return list_recurse
            else:
                self._push()
                return self._getref

        # We handle tuples and sets by encoding them in a "(tuple|set)dict"
        if util.is_tuple(obj):
            if not self.unpicklable:
                return list_recurse
            return lambda obj: {tags.TUPLE: [self.flatten(v) for v in obj]}

        if util.is_set(obj):
            if not self.unpicklable:
                return list_recurse
            return lambda obj: {tags.SET: [self.flatten(v) for v in obj]}

        if util.is_dictionary(obj):
            return self._flatten_dict_obj

        if util.is_type(obj):
            return _mktyperef

        if util.is_object(obj):
            return self._ref_obj_instance

        # else, what else? (methods, functions, old style classes...)
        return None

    def _ref_obj_instance(self, obj):
        """Reference an existing object or flatten if new
        """
        if self._mkref(obj):
            # We've never seen this object so return its
            # json representation.
            return self._flatten_obj_instance(obj)
        # We've seen this object before so place an object
        # reference tag in the data. This avoids infinite recursion
        # when processing cyclical objects.
        return self._getref(obj)

    def _flatten_obj_instance(self, obj):
        """Recursively flatten an instance and return a json-friendly dict
        """
        data = {}
        has_class = hasattr(obj, '__class__')
        has_dict = hasattr(obj, '__dict__')
        has_slots = not has_dict and hasattr(obj, '__slots__')
        has_getstate = has_dict and hasattr(obj, '__getstate__')
        has_getstate_support = has_getstate and hasattr(obj, '__setstate__')

        HandlerClass = handlers.BaseHandler._registry.get(type(obj))

        if has_class and not util.is_module(obj):
            module, name = _getclassdetail(obj)
            if self.unpicklable:
                data[tags.OBJECT] = '%s.%s' % (module, name)
            # Check for a custom handler
            if HandlerClass:
                handler = HandlerClass(self)
                flat_obj = handler.flatten(obj, data)
                self._mkref(flat_obj)
                return flat_obj

        if util.is_module(obj):
            if self.unpicklable:
                data[tags.REPR] = '%s/%s' % (obj.__name__,
                                             obj.__name__)
            else:
                data = unicode(obj)
            return data

        if util.is_dictionary_subclass(obj):
            return self._flatten_dict_obj(obj, data)

        if has_dict:
            # Support objects that subclasses list and set
            if util.is_collection_subclass(obj):
                return self._flatten_collection_obj(obj, data)

            # Support objects with __getstate__(); this ensures that
            # both __setstate__() and __getstate__() are implemented
            if has_getstate_support:
                state = self.flatten(obj.__getstate__())
                if self.unpicklable:
                    data[tags.STATE] = state
                else:
                    data = state
                return data

            # hack for zope persistent objects; this unghostifies the object
            getattr(obj, '_', None)
            return self._flatten_dict_obj(obj.__dict__, data)

        if util.is_collection_subclass(obj):
            return self._flatten_collection_obj(obj, data)

        if util.is_noncomplex(obj):
            return [self.flatten(v) for v in obj]

        if has_slots:
            return self._flatten_newstyle_with_slots(obj, data)

    def _flatten_dict_obj(self, obj, data=None):
        """Recursively call flatten() and return json-friendly dict
        """
        if data is None:
            data = obj.__class__()

        flatten = self._flatten_key_value_pair
        for k, v in sorted(obj.items(), key=operator.itemgetter(0)):
            flatten(k, v, data)

        # the collections.defaultdict protocol
        if hasattr(obj, 'default_factory') and callable(obj.default_factory):
            flatten('default_factory', obj.default_factory, data)

        return data

    def _flatten_newstyle_with_slots(self, obj, data):
        """Return a json-friendly dict for new-style objects with __slots__.
        """
        for k in obj.__slots__:
            self._flatten_key_value_pair(k, getattr(obj, k), data)
        return data

    def _flatten_key_value_pair(self, k, v, data):
        """Flatten a key/value pair into the passed-in dictionary."""
        if not util.is_picklable(k, v):
            return data
        if not isinstance(k, (str, unicode)):
            try:
                k = repr(k)
            except:
                k = unicode(k)
        data[k] = self.flatten(v)
        return data

    def _flatten_collection_obj(self, obj, data):
        """Return a json-friendly dict for a collection subclass."""
        if hasattr(obj, '__dict__'):
            self._flatten_dict_obj(obj.__dict__, data)
        value = [self.flatten(v) for v in obj]
        if self.unpicklable:
            data[tags.SEQ] = value
        else:
            return value
        return data

def _mktyperef(obj):
    """Return a typeref dictionary.  Used for references.

    >>> from jsonpickle import tags
    >>> _mktyperef(AssertionError)[tags.TYPE].rsplit('.', 1)[0]
    'exceptions'

    >>> _mktyperef(AssertionError)[tags.TYPE].rsplit('.', 1)[-1]
    'AssertionError'
    """
    return {tags.TYPE: '%s.%s' % (obj.__module__, obj.__name__)}

def _getclassdetail(obj):
    """Helper class to return the class of an object.

    >>> class Example(object): pass
    >>> _getclassdetail(Example())
    ('jsonpickle.pickler', 'Example')
    >>> _getclassdetail(25)
    ('__builtin__', 'int')
    >>> _getclassdetail(None)
    ('__builtin__', 'NoneType')
    >>> _getclassdetail(False)
    ('__builtin__', 'bool')
    """
    cls = obj.__class__
    module = getattr(cls, '__module__')
    name = getattr(cls, '__name__')
    return module, name
