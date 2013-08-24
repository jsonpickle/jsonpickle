import sys

# True if we are running on Python 3.
PY3 = sys.version_info[0] == 3

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
except NameError:
    long = int

try:
    unichr = unichr
except NameError:
    unichr = chr
