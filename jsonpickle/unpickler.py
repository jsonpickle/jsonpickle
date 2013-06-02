# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett (john -at- paulett.org)
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

import operator
import sys
import jsonpickle.util as util
import jsonpickle.tags as tags
import jsonpickle.handlers as handlers
from jsonpickle.compat import set


class Unpickler(object):
    def __init__(self):
        ## The current recursion depth
        self._depth = 0
        ## Maps reference names to object instances
        self._namedict = {}
        ## The namestack grows whenever we recurse into a child object
        self._namestack = []

        ## Maps objects to their index in the _objs list
        self._obj_to_idx = {}
        self._objs = []

    def _reset(self):
        """Resets the object's internal state.
        """
        self._namedict = {}
        self._namestack = []
        self._obj_to_idx = {}
        self._objs = []

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

        if has_tag(obj, tags.ID):
            return self._pop(self._objs[obj[tags.ID]])

        # Backwards compatibility
        if has_tag(obj, tags.REF):
            return self._pop(self._namedict.get(obj[tags.REF]))

        if has_tag(obj, tags.TYPE):
            typeref = loadclass(obj[tags.TYPE])
            if not typeref:
                return self._pop(obj)
            return self._pop(typeref)

        # Backwards compatibility
        if has_tag(obj, tags.REPR):
            obj = loadrepr(obj[tags.REPR])
            return self._pop(self._mkref(obj))

        if has_tag(obj, tags.OBJECT):
            cls = loadclass(obj[tags.OBJECT])
            if not cls:
                return self._pop(self._mkref(obj))

            # check custom handlers
            HandlerClass = handlers.BaseHandler._registry.get(cls)
            if HandlerClass:
                handler = HandlerClass(self)
                instance = handler.restore(obj)
                return self._pop(self._mkref(instance))

            factory = loadfactory(obj)
            args = getargs(obj)
            if args:
                args = self.restore(args)
            try:
                if hasattr(cls, '__new__'):
                    # new style classes
                    if factory:
                        instance = cls.__new__(cls, factory, *args)
                        instance.default_factory = factory
                    else:
                        instance = cls.__new__(cls, *args)
                else:
                    instance = object.__new__(cls)
            except TypeError:
                # old-style classes
                try:
                    instance = cls()
                except TypeError:
                    # fail gracefully if the constructor requires arguments
                    return self._pop(self._mkref(obj))

            # Add to the instance table to allow being referenced by a
            # downstream object
            self._mkref(instance)

            if isinstance(instance, tuple):
                return self._pop(instance)

            if hasattr(instance, '__setstate__') and has_tag(obj, tags.STATE):
                state = self.restore(obj[tags.STATE])
                instance.__setstate__(state)
                return self._pop(instance)

            for k, v in sorted(obj.items(), key=operator.itemgetter(0)):
                # ignore the reserved attribute
                if k in tags.RESERVED:
                    continue
                self._namestack.append(k)
                # step into the namespace
                value = self.restore(v)
                if (util.is_noncomplex(instance) or
                        util.is_dictionary_subclass(instance)):
                    instance[k] = value
                else:
                    setattr(instance, k, value)
                # step out
                self._namestack.pop()

            # Handle list and set subclasses
            if has_tag(obj, tags.SEQ):
                if hasattr(instance, 'append'):
                    for v in obj[tags.SEQ]:
                        instance.append(self.restore(v))
                if hasattr(instance, 'add'):
                    for v in obj[tags.SEQ]:
                        instance.add(self.restore(v))

            return self._pop(instance)

        if util.is_list(obj):
            parent = []
            self._mkref(parent)
            children = [self.restore(v) for v in obj]
            parent.extend(children)
            return self._pop(parent)

        if has_tag(obj, tags.TUPLE):
            return self._pop(tuple([self.restore(v)
                                    for v in obj[tags.TUPLE]]))

        if has_tag(obj, tags.SET):
            return self._pop(set([self.restore(v)
                                  for v in obj[tags.SET]]))

        if util.is_dictionary(obj):
            data = {}
            for k, v in sorted(obj.items(), key=operator.itemgetter(0)):
                self._namestack.append(k)
                data[k] = self.restore(v)
                self._namestack.pop()

            return self._pop(data)

        return self._pop(obj)

    def _refname(self):
        """Calculates the name of the current location in the JSON stack.

        This is called as jsonpickle traverses the object structure to
        create references to previously-traversed objects.  This allows
        cyclical data structures such as doubly-linked lists.
        jsonpickle ensures that duplicate python references to the same
        object results in only a single JSON object definition and
        special reference tags to represent each reference.

        >>> u = Unpickler()
        >>> u._namestack = []
        >>> u._refname()
        '/'

        >>> u._namestack = ['a']
        >>> u._refname()
        '/a'

        >>> u._namestack = ['a', 'b']
        >>> u._refname()
        '/a/b'

        """
        return '/' + '/'.join(self._namestack)

    def _mkref(self, obj):
        """
        >>> from jsonpickle._samples import Thing
        >>> thing = Thing('referenced-thing')
        >>> u = Unpickler()
        >>> u._mkref(thing)
        Thing("referenced-thing")

        >>> u._objs[0]
        Thing("referenced-thing")

        """
        obj_id = id(obj)
        try:
            self._obj_to_idx[obj_id]
        except KeyError:
            self._obj_to_idx[obj_id] = len(self._objs)
            self._objs.append(obj)
            # Backwards compatibility: old versions of jsonpickle
            # produced "py/ref" references.
            self._namedict[self._refname()] = obj
        return obj


def loadclass(module_and_name):
    """Loads the module and returns the class.

    >>> loadclass('jsonpickle._samples.Thing')
    <class 'jsonpickle._samples.Thing'>

    >>> loadclass('example.module.does.not.exist.Missing')


    >>> loadclass('samples.MissingThing')


    """
    try:
        module, name = module_and_name.rsplit('.', 1)
        __import__(module)
        return getattr(sys.modules[module], name)
    except:
        return None


def loadfactory(obj):
    try:
        default_factory = obj['default_factory']
    except KeyError:
        return None
    try:
        type_tag = default_factory[tags.TYPE]
    except:
        return None

    typeref = loadclass(type_tag)
    if typeref:
        del obj['default_factory']
        return typeref

    return None


def getargs(obj):
    try:
        seq_list = obj[tags.SEQ]
        obj_dict = obj[tags.OBJECT]
    except KeyError:
        return []
    typeref = loadclass(obj_dict)
    if not typeref:
        return []
    if hasattr(typeref, '_fields'):
        if len(typeref._fields) == len(seq_list):
            return seq_list
    return []


def loadrepr(reprstr):
    """Returns an instance of the object from the object's repr() string.
    It involves the dynamic specification of code.

    >>> loadrepr('jsonpickle._samples/jsonpickle._samples.Thing("json")')
    Thing("json")

    """
    module, evalstr = reprstr.split('/')
    mylocals = locals()
    localname = module
    if '.' in localname:
        localname = module.split('.', 1)[0]
    mylocals[localname] = __import__(module)
    return eval(evalstr)


def has_tag(obj, tag):
    """Helper class that tests to see if the obj is a dictionary
    and contains a particular key/tag.

    >>> obj = {'test': 1}
    >>> has_tag(obj, 'test')
    True
    >>> has_tag(obj, 'fail')
    False

    >>> has_tag(42, 'fail')
    False

    """
    return type(obj) is dict and tag in obj
