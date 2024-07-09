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
and `JSON <http://json.org/>`_.  jsonpickle builds upon existing JSON
encoders, such as simplejson, json, and ujson.

.. warning::

   jsonpickle can execute arbitrary Python code.

   Please see the Security section for more details.


For complete documentation, please visit the
`jsonpickle documentation <http://jsonpickle.readthedocs.io/>`_.

Bug reports and merge requests are encouraged at the
`jsonpickle repository on github <https://github.com/jsonpickle/jsonpickle>`_.

Usage
=====
The following is a very simple example of how one can use jsonpickle in their scripts/projects. Note the usage of jsonpickle.encode and decode, and how the data is written/encoded to a file and then read/decoded from the file.

.. code-block:: python

    import jsonpickle
    from dataclasses import dataclass
   
    @dataclass
    class Example:
        data: str
   
   
    ex = Example("value1")
    encoded_instance = jsonpickle.encode(ex)
    assert encoded_instance == '{"py/object": "__main__.Example", "data": "value1"}'
   
    with open("example.json", "w+") as f:
        f.write(encoded_instance)
   
    with open("example.json", "r+") as f:
        written_instance = f.read()
        decoded_instance = jsonpickle.decode(written_instance)
    assert decoded_instance == ex

For more examples, see the `examples directory on GitHub <https://github.com/jsonpickle/jsonpickle/tree/main/examples>`_ for example scripts. These can be run on your local machine to see how jsonpickle works and behaves, and how to use it. Contributions from users regarding how they use jsonpickle are welcome!


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


Numpy/Pandas Support
====================

jsonpickle includes built-in numpy and pandas extensions.  If you would
like to encode sklearn models, numpy arrays, pandas DataFrames, and other
numpy/pandas-based data, then you must enable the numpy and/or pandas
extensions by registering their handlers::

    >>> import jsonpickle.ext.numpy as jsonpickle_numpy
    >>> import jsonpickle.ext.pandas as jsonpickle_pandas
    >>> jsonpickle_numpy.register_handlers()
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

The testing requirements are specified in `setup.cfg`.
It is recommended to create a virtualenv and run tests from within the
virtualenv.::

        python3 -mvenv env3
        source env3/bin/activate
        pip install --editable '.[dev]'
        make test

You can also use a tool such as `vx <https://github.com/davvid/vx/>`_
to activate the virtualenv without polluting your shell environment::

        python3 -mvenv env3
        vx env3 pip install --editable '.[dev]'
        vx env3 make test

If you can't use a venv, you can install the testing packages as follows::

        pip install .[testing]

`jsonpickle` supports multiple Python versions, so using a combination of
multiple virtualenvs and `tox` is useful in order to catch compatibility
issues when developing.

GPG Signing
===========

Unfortunately, while versions of jsonpickle before 3.0.1 should still be signed, GPG signing support was removed from PyPi (https://blog.pypi.org/posts/2023-05-23-removing-pgp/) back in May 2023.

License
=======

Licensed under the BSD License. See COPYING for details.
