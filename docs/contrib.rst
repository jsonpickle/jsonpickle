==========================
Contributing to jsonpickle
==========================

We welcome contributions from everyone.  Please fork jsonpickle on
`github <http://github.com/jsonpickle/jsonpickle>`_.

Get the Code
============

.. _jsonpickle-contrib-checkout:

::

    git clone https://github.com/jsonpickle/jsonpickle.git

Run the Test Suite
==================

All tests should pass before before code is merged into the main repository.
If you are contributing an addition or a change in behavior, we ask that you
document the change in the form of test cases.

.. _tox: https://tox.readthedocs.io/

The test suite is most readily run with the `tox`_ testing tool.
Once installed, run the test suite against the default Python::

    tox

It is recommended that you install at least two Python3
interpreters for use by tox_. To test against Python 3.10 and 3.11::

    tox -e py310,py311

The jsonpickle test suite uses several JSON encoding libraries as well as
several libraries for sample objects. To create an environment to test
against these libs::

    tox -e libs

To test against these libs on Python 3.10::

    tox -e py310-libs

To create the environment without running tests::

    tox -e libs --notest

Now you may experiment and interact with jsonpickle under development
from the virtualenv at ``.tox/libs/{bin/Scripts}/python``.


Generate Documentation
======================

Generating the documentation_ is not necessary when contributing.
To build the docs::

    tox -e docs

Now docs are available in ``build/html``.

If you wish to browse the documentation, use Python's :mod:`http.server`
to host them at http://localhost:8000::

    python -m http.server -d build/html

.. _documentation: http://jsonpickle.github.com
.. _Sphinx: http://sphinx.pocoo.org
