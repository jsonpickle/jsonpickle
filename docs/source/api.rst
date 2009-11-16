.. _jsonpickle-api:

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


:mod:`jsonpickle.handlers` -- Custom Serialization Handlers
===========================================================

The jsonpickle.handlers.registry allows plugging in custom
serialization handlers at run-time.  This is useful when
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


:mod:`jsonpickle.util` -- Helper functions
------------------------------------------

.. automodule:: jsonpickle.util
    :members:
    :undoc-members:
