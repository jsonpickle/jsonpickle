.. jsonpickle documentation master file, created by sphinx-quickstart on Thu Nov 27 14:21:06 2008.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. contents::

jsonpickle Usage
================

.. automodule:: jsonpickle


Download & Install
==================

The easiest way to get jsonpickle is via PyPi_::
    
    $ easy_install jsonpickle
    
You can also download_ or :ref:`checkout <jsonpickle-contrib-checkout>` the 
latest code::

    $ python setup.py install   
    
.. _PyPi: http://pypi.python.org/
.. _download: http://code.google.com/p/jsonpickle/downloads/list


API Reference
=============

.. toctree::
   :maxdepth: 3
   
   api

Contributing
============

.. _jsonpickle-contrib-checkout:

::

    svn checkout http://jsonpickle.googlecode.com/svn/trunk/ jsonpickle

   
Contact
=======

Please join our `mailing list <http://groups.google.com/group/jsonpickle>`_.  
You can send email to *jsonpickle@googlegroups.com*.  

Check http://blog.7oars.com and http://code.google.com/p/jsonpickle/ for 
project updates.


Authors
=======

 * John Paulett (john -at- 7oars.com)
 * David Aguilar (davvid -at- gmail.com)

Change Log
==========

Version 0.2.0 - January 10, 2009
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
    * Added long as basic primitive (thanks Adam Fisk)
    * Prefer python-cjson to simplejson, if available
    * Major API change, use python-cjson's decode/encode instead of
      simplejson's load/loads/dump/dumps
    * Added benchmark.py to compare simplejson and python-cjson

Version 0.0.5 - July 21, 2008
    * Changed prefix of special fields to conform with CouchDB 
      requirements (Thanks Dean Landolt). Break backwards compatibility.
    * Moved to Google Code subversion
    * Fixed unit test imports

Version 0.0.3
    * Convert back to setup.py from pavement.py (issue found by spidaman) 

Version 0.0.2
    * Handle feedparser's FeedParserDict
    * Converted project to Paver
    * Restructured directories
    * Increase test coverage

Version 0.0.1
Initial release

   
License
=======

jsonpickle is provided under a 
`New BSD license <http://code.google.com/p/jsonpickle/source/browse/trunk/src/COPYING>`_, 

Copyright (C) 2008-2009 John Paulett (john -at- 7oars.com)

