Change Log
==========
Version 0.7.1 - May 6, 2014
------------------------------
    * Added support for Python 3.4.
    * Added support for :class:`posix.stat_result`.

Version 0.7.0 - March 15, 2014
------------------------------

    * Added ``handles`` decorator to :class:`jsonpickle.handlers.BaseHandler`,
      enabling simple declaration of a handler for a class.
    * `__getstate__()` and `__setstate__()` are now honored
      when pickling objects that subclass :class:`dict`.
    * jsonpickle can now serialize :class:`collections.Counter` objects.
    * Object references are properly handled when using integer keys.
    * Object references are now supported when using custom handlers.
    * Decimal objects are supported in Python 3.
    * jsonpickle's "fallthrough-on-error" behavior can now be disabled.
    * Simpler API for registering custom handlers.
    * A new "safe-mode" is provided which avoids eval().
      Backwards-compatible deserialization of repr-serialized objects
      is disabled in this mode.  e.g. `decode(string, safe=True)`

Version 0.6.1 - August 25, 2013
-------------------------------

    * Python 3.2 support, and additional fixes for Python 3.

Version 0.6.0 - August 24, 2013
-------------------------------

    * Python 3 support!
    * :class:`time.struct_time` is now serialized using the built-in
      :class:`jsonpickle.handlers.SimpleReduceHandler`.

Version 0.5.0 - August 22, 2013
-------------------------------

    * Non-string dictionary keys (e.g. ints, objects) are now supported
      by passing `keys=True` to :func:`jsonpickle.encode` and
      :func:`jsonpickle.decode`.
    * We now support namedtuple, deque, and defaultdict.
    * Datetimes with timezones are now fully supported.
    * Better support for complicated structures e.g.
      datetime inside dicts.
    * jsonpickle added support for references and cyclical data structures
      in 0.4.0.  This can be disabled by passing `make_refs=False` to
      :func:`jsonpickle.encode`.

Version 0.4.0 - June 21, 2011
-----------------------------

    * Switch build from setuptools to distutils
    * Consistent dictionary key ordering
    * Fix areas with improper support for unpicklable=False
    * Added support for cyclical data structures
      (`#16 <https://github.com/jsonpickle/jsonpickle/issues/16>`_).
    * Experimental support for  `jsonlib <http://pypi.python.org/pypi/jsonlib/>`_
      and `py-yajl <http://github.com/rtyler/py-yajl/>`_ backends.
    * New contributers David K. Hess and Alec Thomas

    .. warning::

        To support cyclical data structures
        (`#16 <https://github.com/jsonpickle/jsonpickle/issues/16>`_),
        the storage format has been modified.  Efforts have been made to
        ensure backwards-compatibility.  jsonpickle 0.4.0 can read data
        encoded by jsonpickle 0.3.1, but earlier versions of jsonpickle may be
        unable to read data encoded by jsonpickle 0.4.0.


Version 0.3.1 - December 12, 2009
---------------------------------

    * Include tests and docs directories in sdist for distribution packages.

Version 0.3.0 - December 11, 2009
---------------------------------

    * Officially migrated to git from subversion. Project home now at
      `<http://jsonpickle.github.com/>`_. Thanks to Michael Jone's
      `sphinx-to-github <http://github.com/michaeljones/sphinx-to-github>`_.
    * Fortified jsonpickle against common error conditions.
    * Added support for:

     * List and set subclasses.
     * Objects with module references.
     * Newstyle classes with `__slots__`.
     * Objects implementing `__setstate__()` and `__getstate__()`
       (follows the :mod:`pickle` protocol).

    * Improved support for Zope objects via pre-fetch.
    * Support for user-defined serialization handlers via the
      jsonpickle.handlers registry.
    * Removed cjson support per John Millikin's recommendation.
    * General improvements to style, including :pep:`257` compliance and
      refactored project layout.
    * Steps towards Python 2.3 and Python 3 support.
    * New contributors Dan Buch and Ian Schenck.
    * Thanks also to Kieran Darcy, Eoghan Murray, and Antonin Hildebrand
      for their assistance!

Version 0.2.0 - January 10, 2009
--------------------------------

    * Support for all major Python JSON backends (including json in Python 2.6,
      simplejson, cjson, and demjson)
    * Handle several datetime objects using the repr() of the objects
      (Thanks to Antonin Hildebrand).
    * Sphinx documentation
    * Added support for recursive data structures
    * Unicode dict-keys support
    * Support for Google App Engine and Django
    * Tons of additional testing and bug reports (Antonin Hildebrand, Sorin,
      Roberto Saccon, Faber Fedor,
      `FirePython <http://github.com/darwin/firepython/tree/master>`_, and
      `Joose <http://code.google.com/p/joose-js/>`_)

Version 0.1.0 - August 21, 2008
-------------------------------

    * Added long as basic primitive (thanks Adam Fisk)
    * Prefer python-cjson to simplejson, if available
    * Major API change, use python-cjson's decode/encode instead of
      simplejson's load/loads/dump/dumps
    * Added benchmark.py to compare simplejson and python-cjson

Version 0.0.5 - July 21, 2008
-----------------------------

    * Changed prefix of special fields to conform with CouchDB
      requirements (Thanks Dean Landolt). Break backwards compatibility.
    * Moved to Google Code subversion
    * Fixed unit test imports

Version 0.0.3
-------------

    * Convert back to setup.py from pavement.py (issue found by spidaman)

Version 0.0.2
-------------

    * Handle feedparser's FeedParserDict
    * Converted project to Paver
    * Restructured directories
    * Increase test coverage

Version 0.0.1
-------------

    Initial release
