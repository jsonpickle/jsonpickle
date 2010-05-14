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

The jsonpickle test suite uses several JSON encoding libraries as well as 
several libraries for sample objects.  To simplify the process of setting up
these libraries we recommend creating a virtualenv_ and using a pip_ 
requirements file to install the dependencies.  In the base jsonpickle 
directory::

    # create a virtualenv that is completely isolated from the 
    # site-wide python install
    virtualenv --no-site-packages env

    # activate the virtualenv
    source env/bin/activate

    # install pip
    easy_install pip

    # use pip to install the dependencies listed in the requirements file
    env/bin/pip install --upgrade -r tests/test-req.txt

To run the suite, simply envoke :file:`tests/runtests.py`::

    $ tests/runtests.py
    test_None (util_tests.IsPrimitiveTestCase) ... ok
    test_bool (util_tests.IsPrimitiveTestCase) ... ok
    test_dict (util_tests.IsPrimitiveTestCase) ... ok
    test_float (util_tests.IsPrimitiveTestCase) ... ok
    ...


.. _virtualenv: http://pypi.python.org/pypi/virtualenv
.. _pip: http://pypi.python.org/pypi/pip
