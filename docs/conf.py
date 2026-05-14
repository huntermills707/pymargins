import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'pymargins'
copyright = '2026'
author = ''
release = '0.0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'myst_nb',
    'sphinx_design',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst-nb',
    '.ipynb': 'myst-nb',
}

napoleon_numpy_docstring = True
napoleon_google_docstring = False

autosummary_generate = True
autodoc_member_order = 'bysource'

# MyST + notebook execution
myst_enable_extensions = [
    'dollarmath',
    'amsmath',
    'colon_fence',
    'deflist',
]
myst_heading_anchors = 3

# Notebooks: scaffold ships with execution disabled so the docs build
# does not require every model backend (statsmodels, linearmodels,
# lifelines, JAX) to be installed. Flip to 'cache' once notebooks are
# pinned to a known-good environment; then stale outputs become a build
# error rather than a silent drift.
nb_execution_mode = 'off'
nb_execution_timeout = 180
nb_execution_raise_on_error = True

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'statsmodels': ('https://www.statsmodels.org/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'jax': ('https://docs.jax.dev/en/latest/', None),
    'lifelines': ('https://lifelines.readthedocs.io/en/latest/', None),
    'linearmodels': ('https://bashtage.github.io/linearmodels/', None),
}

exclude_patterns = [
    '_build',
    'jupyter_execute',
    '**/jupyter_execute',
    'tutorials/README.md',
    'howto/README.md',
    'explanations/README.md',
    '**/README.md',
]

html_theme = 'alabaster'
html_static_path = ['_static']
html_css_files = ['custom.css']
