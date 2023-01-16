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

# pylint: disable=invalid-name

import os
import sys
import shutil
import datetime

sys.path.insert(0, os.path.abspath("platipy"))

html_theme = "furo"

# -- Project information -----------------------------------------------------

project = "PlatiPy"
year = datetime.datetime.now().year
copyright = f"{year}, Ingham Institute, University of New South Wales, University of Sydney"
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
    "m2r2",
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


def setup(app):
    print("Adding custom css...")
    app.add_css_file("_static/custom.css")  # may also be an URL


shutil.rmtree("_examples", ignore_errors=True)
os.mkdir("_examples")
shutil.copy("../examples/visualise.ipynb", "_examples/visualise.ipynb")
shutil.copy("../examples/dvh_analysis.ipynb", "_examples/dvh_analysis.ipynb")
shutil.copy("../examples/atlas_segmentation.ipynb", "_examples/atlas_segmentation.ipynb")
shutil.copy("../examples/contour_comparison.ipynb", "_examples/contour_comparison.ipynb")
shutil.copy("../examples/bronchus_segmentation.ipynb", "_examples/bronchus_segmentation.ipynb")

shutil.rmtree("site/assets", ignore_errors=True)
os.makedirs("site", exist_ok=True)
shutil.copytree("../assets", "site/assets")
