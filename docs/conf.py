import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "pymargins"
copyright = "2026"
author = "Hunter Mills"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_nb",
    "sphinx_design",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

napoleon_numpy_docstring = True
napoleon_google_docstring = False

autosummary_generate = True
autodoc_member_order = "bysource"

# MyST + notebook execution
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

# Notebook execution: 'auto' executes notebooks that lack outputs,
# 'cache' reuses cached outputs when inputs haven't changed (fastest
# for CI), and 'off' disables execution entirely.  Override with the
# PYMARGINS_DOCS_EXEC environment variable.
# Default to 'cache' on ReadTheDocs so already-cached notebooks don't
# re-execute on every build.
_default_exec_mode = "cache" if os.getenv("READTHEDOCS") else "auto"
nb_execution_mode = os.getenv("PYMARGINS_DOCS_EXEC", _default_exec_mode)
nb_execution_cache_path = os.path.join(os.path.dirname(__file__), ".jupyter_cache")
nb_execution_timeout = 180
nb_execution_raise_on_error = True
nb_execution_show_tb = True

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "lifelines": ("https://lifelines.readthedocs.io/en/latest/", None),
    "linearmodels": ("https://bashtage.github.io/linearmodels/", None),
}

exclude_patterns = [
    "_build",
    "jupyter_execute",
    "**/jupyter_execute",
    ".jupyter_cache",
    "**/.jupyter_cache",
    "tutorials/README.md",
    "howto/README.md",
    "explanations/README.md",
    "**/README.md",
]

html_theme = "alabaster"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "github_user": "huntermills707",
    "github_repo": "pymargins",
    "github_button": True,
    "github_type": "star",
    "description": "Expert-mode marginal effects for Python.",
    "extra_nav_links": {
        "GitHub repository": "https://github.com/huntermills707/pymargins",
        "Issue tracker": "https://github.com/huntermills707/pymargins/issues",
        "Changelog": "https://github.com/huntermills707/pymargins/blob/main/CHANGELOG.md",
        "PyPI": "https://pypi.org/project/pymargins/",
    },
}
