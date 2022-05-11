from __future__ import absolute_import, division, unicode_literals

import base64
import sys
import types

PY_MAJOR = sys.version_info[0]

class_types = (type,)
iterator_types = (type(iter('')),)

import builtins
import queue
from base64 import decodebytes, encodebytes
from collections.abc import Iterator as abc_iterator

string_types = (str,)
numeric_types = (int, float)
ustr = str


def iterator(class_):
    # TODO: Replace all instances of this
    return class_
