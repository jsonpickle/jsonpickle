#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2008 John Paulett (john -at- paulett.org)
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution.

from distutils.core import setup
import jsonpickle as _jsonpickle
import sys

SETUP_ARGS = dict(
    name="jsonpickle",
    version=_jsonpickle.__version__,
    description="Python library for serializing any "
                "arbitrary object graph into JSON",
    long_description = _jsonpickle.__doc__,
    author="John Paulett",
    author_email="john -at- paulett.org",
    url="http://jsonpickle.github.com/",
    license="BSD",
    platforms=['POSIX', 'Windows'],
    keywords=['json pickle', 'json', 'pickle', 'marshal',
              'serialization', 'JavaScript Object Notation'],
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Development Status :: 4 - Beta",
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

def _check_dependencies():
    # check to see if any of the supported backends is installed
    backends = _jsonpickle.SUPPORTED_BACKENDS
    if not any([_jsonpickle.util.is_installed(module) for module in backends]):
        print >> sys.stderr, ('No supported JSON backend found. '
                              'Must install one of %s' % (', '.join(backends)))
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main())
