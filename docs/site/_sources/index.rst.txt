.. PlatiPy documentation master file, created by
   sphinx-quickstart on Thu Nov 19 00:32:10 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#####################
PlatiPy documentation
#####################


.. toctree::
   :maxdepth: 3
   :caption: Contents:

   index

PlatiPy is a **P**\ rocessing **L**\ ibrary and **A**\ nalysis **T**\ oolkit for Medical 
**I**\ maging in **Py**\ thon. It contains a collections of tools and scripts useful for processing
and analysis of medical images.

Generally these tools are accesible via a Command Line Interface or by calling them in a Python
script directly. Examples are provided in Jupyter Notebooks to demonstrate how to use them from
within Python code.

************
Installation
************

Currently this repository is not available in the official Python repositories. However, you can
install it with pip using the following command:

``pip install git+git://github.com/pyplati/platipy``

If you have already installed the library, you can install the latest updates using:

``pip install --upgrade git+git://github.com/pyplati/platipy``

***************
Getting Started
***************

Command Line Interface (CLI)
============================

Once you have installed PlatiPy, the ``platipy`` command should be available in your environment.
Try the following command to check that is is working:

``platipy --help``

This lists the different tools which are available via the command line interface. Learn more about
these tools and how to use them here: :doc:`here <cli>`.

Python
======

Jupyter Notebooks
-----------------

The following Jupyter Notebooks are provided to demonstrate how to use some of the PlatiPy
functionality from your Python code:

* `Nifti to RTStruct <https://github.com/pyplati/platipy/blob/master/platipy/dicom/nifti_to_rtstruct/convert_sample.ipynb>`_
* `RTStruct to Nifti <https://github.com/pyplati/platipy/blob/master/platipy/dicom/rtstruct_to_nifti/convert_sample.ipynb>`_
* `Visualise Segmentation <https://github.com/pyplati/platipy/blob/master/platipy/imaging/visualisation/examples/visualise_segmentation.ipynb>`_

Module Reference
----------------

You can find more detailed documentation on how to use the different classes and functions found in
PlatiPy :doc:`here <gen/modules>`.

*****
Links
*****

* :doc:`cli`
* :doc:`PlatiPy Modules <gen/modules>`
