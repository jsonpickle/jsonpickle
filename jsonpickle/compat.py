# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, unicode_literals
import sys


__all__ = ('bytes', 'set', 'unicode', 'long', 'unichr', 'queue')

PY_MAJOR = sys.version_info[0]
PY_MINOR = sys.version_info[1]
PY2 = PY_MAJOR == 2
PY3 = PY_MAJOR == 3
PY32 = PY3 and PY_MINOR == 2

try:
    bytes = bytes
except NameError:
    bytes = str

try:
    set = set
except NameError:
    from sets import Set as set
    set = set

try:
    unicode = unicode
except NameError:
    unicode = str

try:
    long = long
    numeric_types = (int, float, long)
except NameError:
    long = int
    numeric_types = (int, float)

try:
    unichr = unichr
except NameError:
    unichr = chr


try:
    # Python3
    import queue
except ImportError:
    # Python2
    import Queue as queue
