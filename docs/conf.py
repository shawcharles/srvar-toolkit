from __future__ import annotations

import os
import sys


sys.path.insert(0, os.path.abspath(".."))

project = "srvar-toolkit"
author = "charles shaw"

try:
    import srvar

    release = srvar.__version__
except Exception:
    release = "0.0.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
]

autosummary_generate = True

source_suffix = {
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"

html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "numpy": ("https://numpy.org/doc/stable/", {}),
    "scipy": ("https://docs.scipy.org/doc/scipy/", {}),
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
