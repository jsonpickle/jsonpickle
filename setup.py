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
try:
    import setuptools as setup_mod
except ImportError:
    import distutils.core as setup_mod

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
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Development Status :: 5 - Stable",
        "Intended Audience :: Developers",
        "Programming Language :: JavaScript",
    ],
    options={'clean': {'all': 1}},
    packages=["jsonpickle"],
)


if __name__ == '__main__':
    setup_mod.setup(**SETUP_ARGS)
