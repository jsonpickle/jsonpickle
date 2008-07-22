# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

class Thing(object):
    def __init__(self, name):
        self.name = name
        self.child = None

class DictSubclass(dict):
    name = 'Test'