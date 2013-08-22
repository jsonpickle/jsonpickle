.. _jsonpickle-api:

==============
jsonpickle API
==============

.. testsetup:: *

    import jsonpickle
    import jsonpickle.pickler
    import jsonpickle.unpickler
    import jsonpickle.handlers
    import jsonpickle.util

.. contents::

:mod:`jsonpickle` -- High Level API
===================================

.. autofunction:: jsonpickle.encode

.. autofunction:: jsonpickle.decode

Choosing and Loading Backends
-----------------------------

jsonpickle allows the user to specify what JSON backend to use
when encoding and decoding. By default, jsonpickle will try to use, in
the following order: `simplejson <http://undefined.org/python/#simplejson>`_,
:mod:`json`, and `demjson <http://deron.meranda.us/python/demjson/>`_.
The prefered backend can be set via :func:`jsonpickle.set_preferred_backend`.
Additional JSON backends can be used via :func:`jsonpickle.load_backend`.

For example, users of `Django <http://www.djangoproject.com/>`_ can use the
version of simplejson that is bundled in Django::

    jsonpickle.load_backend('django.util.simplejson', 'dumps', 'loads', ValueError))
    jsonpickle.set_preferred_backend('django.util.simplejson')

Supported backends:

 * :mod:`json` in Python 2.6+
 * `simplejson <http://undefined.org/python/#simplejson>`_
 * `demjson <http://deron.meranda.us/python/demjson/>`_

Experimental backends:

 * `jsonlib <http://pypi.python.org/pypi/jsonlib/>`_
 * yajl via `py-yajl <http://github.com/rtyler/py-yajl/>`_
 * `ujson <https://pypi.python.org/pypi/ujson/>`_

.. autofunction:: jsonpickle.set_preferred_backend

.. autofunction:: jsonpickle.load_backend

.. autofunction:: jsonpickle.remove_backend

.. autofunction:: jsonpickle.set_encoder_options

Customizing JSON output
-----------------------

jsonpickle supports the standard :mod:`pickle` `__getstate__` and `__setstate__`
protocol for representating object instances.

.. method:: object.__getstate__()

   Classes can further influence how their instances are pickled; if the class
   defines the method :meth:`__getstate__`, it is called and the return state is
   pickled as the contents for the instance, instead of the contents of the
   instance's dictionary.  If there is no :meth:`__getstate__` method, the
   instance's :attr:`__dict__` is pickled.

.. method:: object.__setstate__(state)

   Upon unpickling, if the class also defines the method :meth:`__setstate__`,
   it is called with the unpickled state. If there is no
   :meth:`__setstate__` method, the pickled state must be a dictionary and its
   items are assigned to the new instance's dictionary.  If a class defines both
   :meth:`__getstate__` and :meth:`__setstate__`, the state object needn't be a
   dictionary and these methods can do what they want.


:mod:`jsonpickle.handlers` -- Custom Serialization Handlers
-----------------------------------------------------------

The `jsonpickle.handlers` module allows plugging in custom
serialization handlers at run-time.  This feature is useful when
jsonpickle is unable to serialize objects that are not
under your direct control.

.. automodule:: jsonpickle.handlers
    :members:
    :undoc-members:

Low Level API
=============

Typically this low level functionality is not needed by clients.

:mod:`jsonpickle.pickler` -- Python to JSON
-------------------------------------------

.. automodule:: jsonpickle.pickler
    :members:
    :undoc-members:

:mod:`jsonpickle.unpickler` -- JSON to Python
---------------------------------------------

.. automodule:: jsonpickle.unpickler
    :members:
    :undoc-members:

:mod:`jsonpickle.backend` -- JSON Backend Management
----------------------------------------------------

.. automodule:: jsonpickle.backend
    :members:

:mod:`jsonpickle.util` -- Helper functions
------------------------------------------

.. automodule:: jsonpickle.util
    :members:
    :undoc-members:
