# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett (john -at- paulett.org)
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

from jsonpickle.compat import set


class Thing(object):
    def __init__(self, name):
        self.name = name
        self.child = None

    def __repr__(self):
        return 'samples.Thing("%s")' % self.name


class ThingWithSlots(object):
    __slots__ = ('a', 'b')
    def __init__(self, a, b):
        self.a = a
        self.b = b


class ThingWithProps(object):

    def __init__(self, name='', dogs='reliable', monkies='tricksy'):
        self.name = name
        self._critters = (('dogs', dogs), ('monkies', monkies))

    def _get_identity(self):
        keys = [self.dogs, self.monkies, self.name]
        return hash('-'.join([str(key) for key in keys]))

    identity = property(_get_identity)

    def _get_dogs(self):
        return self._critters[0][1]

    dogs = property(_get_dogs)

    def _get_monkies(self):
        return self._critters[1][1]

    monkies = property(_get_monkies)

    def __getstate__(self):
        out = dict(
            __identity__=self.identity,
            nom=self.name,
            dogs=self.dogs,
            monkies=self.monkies,
        )
        return out

    def __setstate__(self, state_dict):
        self._critters = (('dogs', state_dict.get('dogs')),
                          ('monkies', state_dict.get('monkies')))
        self.name = state_dict.get('nom', '')
        ident = state_dict.get('__identity__')
        if ident != self.identity:
            raise ValueError('expanded object does not match originial state!')

    def __eq__(self, other):
        return self.identity == other.identity


class DictSubclass(dict):
    name = 'Test'


class ListSubclass(list):
    pass


class SetSubclass(set):
    pass


class BrokenReprThing(Thing):
    def __repr__(self):
        raise Exception('%s has a broken repr' % self.name)
    def __str__(self):
        return '<BrokenReprThing "%s">' % self.name


class Node(object):
    def __init__(self, name):
        self._name = name
        self._children = []
        self._parent = None

    def add_child(self, child, index=-1):
        if index == -1:
            index = len(self._children)
        self._children.insert(index, child)
        child._parent = self


class Document(Node):
    def __init__(self, name):
        Node.__init__(self, name)

    def __str__(self):
        ret_str ='Document "%s"\n' %self._name
        for c in self._children:
            ret_str += repr(c)

        return ret_str

    def __repr__(self):
        return self.__str__()

class Section(Node):
    def __init__(self, name):
        Node.__init__(self, name)

    def __str__(self):
        ret_str = 'Section "%s", parent: "%s"\n' % (self._name, self._parent._name)
        for c in self._children:
            ret_str += repr(c)
        return ret_str

    def __repr__(self):
        return self.__str__()


class Question(Node):
    def __init__(self, name):
        Node.__init__(self, name)

    def __str__(self):
        return 'Question "%s", parent: "%s"\n' % (self._name, self._parent._name)

    def __repr__(self):
        return self.__str__()
