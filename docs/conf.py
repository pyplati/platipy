# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath("platipy"))

html_theme = "furo"

# -- Project information -----------------------------------------------------

project = "PlatiPy"
copyright = "2021, Phillip Chlap & Rob Finnegan"
author = "Phillip Chlap & Rob Finnegan"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "site"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_show_sphinx = False

print("Copy example notebooks into docs/_examples")


def all_but_ipynb(directory, contents):
    result = []
    for c in contents:
        if os.path.isfile(os.path.join(directory, c)) and (not c.endswith(".ipynb")):
            result += [c]
    return result


shutil.rmtree("_examples", ignore_errors=True)
os.mkdir("_examples")
shutil.copy("../examples/visualise.ipynb", "_examples/visualise.ipynb")
