#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett (john -at- paulett.org)
# Copyright (C) 2009-2013 David Aguilar (davvid -at- gmail.com)
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

import os
import sys
from distutils.core import setup

here = os.path.dirname(__file__)
version = os.path.join(here, 'jsonpickle', 'version.py')
scope = {}
exec(open(version).read(), scope)

SETUP_ARGS = dict(
    name="jsonpickle",
    version=scope['VERSION'],
    description="Python library for serializing any "
                "arbitrary object graph into JSON",
    long_description =
        "jsonpickle converts complex Python objects to and "
        "from JSON.",
    author="David Aguilar",
    author_email="davvid -at- gmail.com",
    url="http://jsonpickle.github.io/",
    license="BSD",
    platforms=['POSIX', 'Windows'],
    keywords=['json pickle', 'json', 'pickle', 'marshal',
              'serialization', 'JavaScript Object Notation'],
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Development Status :: 5 - Stable",
        "Intended Audience :: Developers",
        "Programming Language :: JavaScript"
    ],
    options={'clean': {'all': 1}},
    packages=["jsonpickle"],
)


def main():
    if sys.argv[1] in ('install', 'build'):
        _check_dependencies()
    setup(**SETUP_ARGS)
    return 0


def _is_installed(module):
    """Tests to see if ``module`` is available on the sys.path

    >>> is_installed('sys')
    True
    >>> is_installed('hopefullythisisnotarealmodule')
    False

    """
    try:
        __import__(module)
        return True
    except ImportError as e:
        return False


def _check_dependencies():
    # check to see if any of the supported backends is installed
    backends = ('json',
                'simplejson',
                'demjson',
                'django.util.simplejson')

    if not any([_is_installed(module) for module in backends]):
        print >> sys.stderr, ('No supported JSON backend found. '
                              'Must install one of %s' % (', '.join(backends)))
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main())
