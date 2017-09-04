==========================
Contributing to jsonpickle
==========================

We welcome contributions from everyone.  Please fork jsonpickle on 
`github <http://github.com/jsonpickle/jsonpickle>`_.

Get the Code
============

.. _jsonpickle-contrib-checkout:

::

    git clone git://github.com/jsonpickle/jsonpickle.git

Run the Test Suite
==================

Before code is pulled into the master jsonpickle branch, all tests should pass.
If you are contributing an addition or a change in behavior, we ask that you
document the change in the form of test cases.

The jsonpickle test suite uses several JSON encoding libraries as well as
several libraries for sample objects.  To simplify the process of setting up
these libraries we recommend creating a virtualenv_ and using a pip_
requirements file to install the dependencies.  In the base jsonpickle
directory::

    # create a virtualenv that is completely isolated from the
    # site-wide python install
    virtualenv --no-site-packages env

    # use pip to install development and testing dependencies
    ./env/bin/pip install --upgrade -r requirements-dev.txt

To run the suite, simply invoke :file:`tests/runtests.py`::

    $ ./env/bin/python tests/runtests.py
    test_None (util_tests.IsPrimitiveTestCase) ... ok
    test_bool (util_tests.IsPrimitiveTestCase) ... ok
    test_dict (util_tests.IsPrimitiveTestCase) ... ok
    test_float (util_tests.IsPrimitiveTestCase) ... ok
    ...

.. _virtualenv: http://pypi.python.org/pypi/virtualenv
.. _pip: http://pypi.python.org/pypi/pip

Testing with Tox
================
jsonpickle supports many versions of Python.  To make it easy to test
mutiple versions of Python you should install the tox_ testing tool,
e.g. on Debian::

    $ sudo apt-get install tox

Once tox_ is installed you can run the test suite against multiple Python
interpreters::

    $ make tox

It is recommended that you install at least one Python2 and one Python3
interpreter for use by tox_.

.. _tox: https://tox.readthedocs.io/

Generate Documentation
======================

You do not need to generate the documentation_ when contributing, though, if 
you are interested, you can generate the docs yourself.  The following requires
Sphinx_ (present if you follow the virtualenv instructions above)::

    # pull down the sphinx-to-github project
    git submodule init
    git submodule update

    cd docs
    make html

If you wish to browse the documentation, use Python's :mod:`SimpleHTTPServer`
to host them at http://localhost:8000::

    cd build/html
    python -m SimpleHTTPServer

.. _documentation: http://jsonpickle.github.com
.. _Sphinx: http://sphinx.pocoo.org
