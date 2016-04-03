#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import setuptools_scm

extensions = [
    'sphinx.ext.autodoc',
    'rst.linker',
]

# General information about the project.
project = 'skeleton'
copyright = '2016 Jason R. Coombs'

# The short X.Y version.
version = setuptools_scm.get_version(root='..', relative_to=__file__)
# The full version, including alpha/beta/rc tags.
release = version

master_doc = 'index'

link_files = {
	'CHANGES.rst': dict(
		using=dict(
			GH='https://github.com',
			project=project,
		),
		replace=[
			dict(
				pattern=r"(Issue )?#(?P<issue>\d+)",
				url='{GH}/jaraco/{project}/issues/{issue}',
			),
			dict(
				pattern=r"^(?m)((?P<scm_version>v?\d+(\.\d+){1,2}))\n[-=]+\n",
				with_scm="{text}\n{rev[timestamp]:%d %b %Y}\n",
			),
		],
	),
}
