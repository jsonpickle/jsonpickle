import os
import sys

try:
    import furo
except ImportError:
    furo = None
try:
    import rst.linker as rst_linker
except ImportError:
    rst_linker = None

# Use the source tree.
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
]

project = 'jsonpickle'
master_doc = 'index'

if furo is not None:
    html_theme = 'furo'

# Link dates and other references in the changelog
if rst_linker is not None:
    extensions += ['rst.linker']

package_url = 'https://github.com/jsonpickle/jsonpickle'
link_files = {
    '../CHANGES.rst': dict(
        using=dict(GH='https://github.com'),
        replace=[
            dict(
                pattern=r'(Issue #|\B#)(?P<issue>\d+)',
                url=package_url + '/issues/{issue}',
            ),
            dict(
                pattern=r'\B\+(?P<pull>\d+)',
                url=package_url + '/pull/{pull}',
            ),
            dict(
                pattern=r'(?m:^((?P<scm_version>v?\d+(\.\d+){1,2}))\n[-=]+\n)',
                with_scm='{text}\n{rev[timestamp]:%d %b %Y}\n',
            ),
            dict(
                pattern=r'PEP[- ](?P<pep_number>\d+)',
                url='https://www.python.org/dev/peps/pep-{pep_number:0>4}/',
            ),
        ],
    )
}

# Be strict about any broken references
nitpicky = True

sphinx_disable = os.environ.get('JSONPICKLE_SPHINX_DISABLE', '')
if 'intersphinx' not in sphinx_disable:
    extensions += ['sphinx.ext.intersphinx']
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'sphinx': ('https://www.sphinx-doc.org/en/stable/', None),
}

# Preserve authored syntax for defaults
autodoc_preserve_defaults = True
