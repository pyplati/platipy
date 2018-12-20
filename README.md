# IMPIT (Ingham Medical Physics Imaging Toolbox)

This toolbox contains tools and scripts used for image processing within the Medical Physics group at Ingham Institute in Liverpool, Australia. The code makes use of ITK and VTK and is written in Python. Jupyter notebooks are provided where possible, mainly for guidance on getting started with using the tools.

## Getting Started

### Requirements

- Python 3.5 or greater
- See requirements.txt for required Python packages

## Contributing

### Git

Create a development branch to work in while you make your changes or implement your new tool. Once complete, head to  [BitBucket to create a pull request](https://bitbucket.org/swscsmedphys/dataweb/pull-requests/new) to merge your changes into the main development branch.

### Style Guide

Python code written in this repository should conform to [PEP 8 Style Guide for Python](https://www.python.org/dev/peps/pep-0008/). You may like to use [*pycodestyle*](https://pycodestyle.readthedocs.io) or [*black*](https://github.com/ambv/black) to ensure that your code conforms to PEP 8 standards.

### Structure

This toolbox is broken up into separate modules. Each module contains several tools, scripts or applications. The following structure should be observed:

- module/
    - data/: Directory containing sample data to be used by the Jupyter notebooks and/or test cases of tools within module
    - tool/
        - **tool.py**: One or several Python scripts
        - **README.md**: Contains description of tool, authors, etc...
        - **Sample.ipynb**: Jupyter notebook demonstrating the basics of using the tool
            - **tests/**: Directory containing test scripts to be run by pytest
        - ...

### Providing command line functionality with *click*

Where possible, the tools and scripts within this toolbox should provide a way to run them from the command line. A simple Python library to provide this functionality is *click*. You simply need to annotate the functions you want to have accessible from the command line. See [the official click documentation](https://click.palletsprojects.com) for an introduction to using *click*.

### Writing unit tests for *pytest*

Automated unit tests are important for code bases to which various authors are contributing, to ensure that their changes don't make any unintended breaking changes to other parts of the code.

This toolbox uses *pytest* as the testing framework. See [the official pytest documentation](https://docs.pytest.org/en/latest/getting-started.html) for an introduction to writing tests with pytest.

Before you submit a pull request, make sure all the tests are passing by running the command:

```
pytest
```

from the root directory of the toolbox.

## Authors

* **Phillip Chlap** - [phillip.chlap@unsw.edu.au](phillip.chlap@unsw.edu.au)

* **Robert Finnegan** - [rfin5459@uni.sydney.edu.au](rfin5459@uni.sydney.edu.au)