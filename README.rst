.. image:: https://img.shields.io/pypi/v/jsonpickle.svg
   :target: `PyPI link`_

.. image:: https://img.shields.io/pypi/pyversions/jsonpickle.svg
   :target: `PyPI link`_

.. _PyPI link: https://pypi.org/project/jsonpickle

.. image:: https://readthedocs.org/projects/jsonpickle/badge/?version=latest
   :target: https://jsonpickle.readthedocs.io/en/latest/?badge=latest

.. image:: https://github.com/jsonpickle/jsonpickle/actions/workflows/test.yml/badge.svg
   :target: https://github.com/jsonpickle/jsonpickle/actions
   :alt: Github Actions

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://github.com/jsonpickle/jsonpickle/blob/main/COPYING
   :alt: BSD


jsonpickle
==========
jsonpickle is a library for the two-way conversion of complex Python objects
and `JSON <http://json.org/>`_.  jsonpickle builds upon the existing JSON
encoders, such as simplejson, json, and ujson.

.. warning::

   jsonpickle can execute arbitrary Python code.

   Please see the Security section for more details.


For complete documentation, please visit the
`jsonpickle documentation <http://jsonpickle.readthedocs.io/>`_.

Bug reports and merge requests are encouraged at the
`jsonpickle repository on github <https://github.com/jsonpickle/jsonpickle>`_.

Why jsonpickle?
===============
Data serialized with python's pickle (or cPickle or dill) is not easily readable outside of python. Using the json format, jsonpickle allows simple data types to be stored in a human-readable format, and more complex data types such as numpy arrays and pandas dataframes, to be machine-readable on any platform that supports json. E.g., unlike pickled data, jsonpickled data stored in an Amazon S3 bucket is indexible by Amazon's Athena.

Security
========

jsonpickle should be treated the same as the
`Python stdlib pickle module <https://docs.python.org/3/library/pickle.html>`_
from a security perspective.

.. warning::

   The jsonpickle module **is not secure**.  Only unpickle data you trust.

   It is possible to construct malicious pickle data which will **execute
   arbitrary code during unpickling**.  Never unpickle data that could have come
   from an untrusted source, or that could have been tampered with.

   Consider signing data with an HMAC if you need to ensure that it has not
   been tampered with.

   Safer deserialization approaches, such as reading JSON directly,
   may be more appropriate if you are processing untrusted data.


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
==============
jsonpickle includes a built-in pandas extension.  If would like to encode
pandas DataFrame or Series objects then you must enable the pandas extension
by registering its handlers::

    >>> import jsonpickle.ext.pandas as jsonpickle_pandas
    >>> jsonpickle_pandas.register_handlers()

Development
===========

Use `make` to run the unit tests::

        make test

`pytest` is used to run unit tests internally.

A `tox` target is provided to run tests using all installed and supported Python versions::

        make tox

`jsonpickle` itself has no dependencies beyond the Python stdlib.
`tox` is required for testing when using the `tox` test runner only.

The testing requirements are specified in `requirements-dev.txt`.
It is recommended to create a virtualenv and run tests from within the
virtualenv, or use a tool such as `vx <https://github.com/davvid/vx/>`_
to activate the virtualenv without polluting the shell environment::

        python3 -mvenv env3x
        vx env3x pip install --requirement requirements-dev.txt
        vx env3x make test

`jsonpickle` supports multiple Python versions, so using a combination of
multiple virtualenvs and `tox` is useful in order to catch compatibility
issues when developing.

GPG Signing
===========
Releases before v3.0.0 are signed with `davvid's key <https://keys.openpgp.org/vks/v1/by-fingerprint/FA41BF59C1B48E8C5F3DA61C8CE26BF4A9F606B0>`_. v3.0.0 and after are likely signed by `Theelx's key <https://github.com/Theelx.gpg>`_. All upcoming releases should be signed by one of these two keys, usually Theelx's key.

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
