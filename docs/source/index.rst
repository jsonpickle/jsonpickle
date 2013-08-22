========================
jsonpickle Documentation
========================

`jsonpickle <http://jsonpickle.github.com>`_ is a Python library for
serialization and deserialization of complex Python objects to and from
JSON.  The standard Python libraries for encoding Python into JSON, such as
the stdlib's json, simplejson, and demjson, can only handle Python
primitives that have a direct JSON equivalent (e.g. dicts, lists, strings,
ints, etc.).  jsonpickle builds on top of these libraries and allows more
complex data structures to be serialized to JSON. jsonpickle is highly
configurable and extendable--allowing the user to choose the JSON backend
and add additional backends.

.. contents::

jsonpickle Usage
================

.. automodule:: jsonpickle


Download & Install
==================

The easiest way to get jsonpickle is via PyPi_ with pip_::

    $ pip install -U jsonpickle

For Python 2.6+, jsonpickle has no required dependencies (it uses the standard
library's :mod:`json` module by default). For Python 2.5 or earlier, you must
install a supported JSON backend (including simplejson or demjson). For example::

    $ pip install simplejson

You can also download or :ref:`checkout <jsonpickle-contrib-checkout>` the
latest code and install from source::

    $ python setup.py install

.. _PyPi: http://pypi.python.org/pypi/jsonpickle
.. _pip: http://pypi.python.org/pypi/pip
.. _download: http://pypi.python.org/pypi/jsonpickle


API Reference
=============

.. toctree::
   :maxdepth: 3

   api

Contributing
============

.. toctree::
   :maxdepth: 3

   contrib

Contact
=======

Please join our `mailing list <http://groups.google.com/group/jsonpickle>`_.
You can send email to *jsonpickle@googlegroups.com*.

Check http://github.com/jsonpickle/jsonpickle for project updates.


Authors
=======

 * John Paulett - john -at- paulett.org - http://github.com/johnpaulett
 * David Aguilar - davvid -at- gmail.com - http://github.com/davvid
 * Dan Buch - http://github.com/meatballhat
 * Ian Schenck - http://github.com/ianschenck
 * David K. Hess - http://github.com/davidkhess
 * Alec Thomas - http://github.com/alecthomas
 * jaraco - https://github.com/jaraco


Change Log
==========

.. toctree::
   :maxdepth: 2

   changelog

License
=======

jsonpickle is provided under a
`New BSD license <https://github.com/jsonpickle/jsonpickle/raw/master/COPYING>`_,

Copyright (C) 2008-2011 John Paulett (john -at- paulett.org)
Copyright (C) 2009-2013 David Aguilar (davvid -at- gmail.com)
