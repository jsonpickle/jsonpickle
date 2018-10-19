.. image:: https://badge.fury.io/py/jsonpickle.svg
   :target: https://badge.fury.io/py/jsonpickle
   :alt: PyPI

.. image:: https://img.shields.io/badge/docs-passing-green.svg
   :target: http://jsonpickle.github.io/
   :alt: docs
   
.. image:: https://travis-ci.org/jsonpickle/jsonpickle.svg?branch=master
   :target: https://travis-ci.org/jsonpickle/jsonpickle
   :alt: travis

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://github.com/jsonpickle/jsonpickle/blob/master/COPYING
   :alt: BSD   

   
jsonpickle
==========
jsonpickle is a library for the two-way conversion of complex Python objects
and `JSON <http://json.org/>`_.  jsonpickle builds upon the existing JSON
encoders, such as simplejson, json, and demjson.

For complete documentation, please visit the
`jsonpickle homepage <http://jsonpickle.github.io/>`_.

Bug reports and merge requests are encouraged at the
`jsonpickle repository on github <https://github.com/jsonpickle/jsonpickle>`_.

jsonpickle supports Python 2.7 and Python 3.4 or greater.

Why jsonpickle?
===============
Data serialized with python's pickle (or cPickle or dill) is not easily readable outside of python. Using the json format, jsonpickle allows simple data types to be stored in a human-readable format, and more complex data types such as numpy arrays and pandas dataframes, to be machine-readable on any platform that supports json. E.g., unlike pickled data, jsonpickled data stored in an Amazon S3 bucket is indexible by Amazon's Athena.

Install
=======

Install from pip for the latest stable release:

::

    pip install jsonpickle

Install from github for the latest changes:

::

    pip install git+https://github.com/jsonpickle/jsonpickle.git

If you have the files checked out for development:

::

    git clone https://github.com/jsonpickle/jsonpickle.git
    cd jsonpickle
    python setup.py develop


Numpy Support
=============
jsonpickle includes a built-in numpy extension.  If would like to encode
sklearn models, numpy arrays, and other numpy-based data then you must
enable the numpy extension by registering its handlers::

    >>> import jsonpickle.ext.numpy as jsonpickle_numpy
    >>> jsonpickle_numpy.register_handlers()

Pandas Support
=============
jsonpickle includes a built-in pandas extension.  If would like to encode
pandas DataFrame or Series objects then you must enable the pandas extension
by registering its handlers::

    >>> import jsonpickle.ext.pandas as jsonpickle_pandas
    >>> jsonpickle_pandas.register_handlers()

jsonpickleJS
============
`jsonpickleJS <https://github.com/cuthbertLab/jsonpickleJS>`_
is a javascript implementation of jsonpickle by Michael Scott Cuthbert.
jsonpickleJS can be extremely useful for projects that have parallel data
structures between Python and Javascript.

License
=======
Licensed under the BSD License. See COPYING for details.
See jsonpickleJS/LICENSE for details about the jsonpickleJS license.
