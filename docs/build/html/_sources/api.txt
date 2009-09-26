.. _jsonpickle-api:

.. testsetup:: *

    from jsonpickle import *
    from jsonpickle.util import *
    from jsonpickle.pickler import *
    from jsonpickle.unpickler import *
    

.. contents::

:mod:`jsonpickle` -- High Level API
===================================

.. autofunction:: jsonpickle.encode

.. autofunction:: jsonpickle.decode


Low Level API
=============

Typically this low level functionality is not needed by clients.

:mod:`jsonpickle.pickler` -- Python to JSON
------------------------------------------

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
