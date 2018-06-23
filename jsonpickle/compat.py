from __future__ import absolute_import, division, unicode_literals
import sys

PY_MAJOR = sys.version_info[0]
PY2 = PY_MAJOR == 2
PY3 = PY_MAJOR == 3

if PY3:
    import queue
    string_types = (str,)
    numeric_types = (int, float)
    ustr = str
else:
    import Queue as queue
    string_types = (basestring,)
    numeric_types = (int, float, long)
    ustr = unicode
