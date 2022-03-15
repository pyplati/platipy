# Contributing

PlatiPy welcomes any and all contributions in the way of new tools/scripts, bug fixes or
documentation. Below you will find information to help you get started.

## Git

Create a branch off of **master** while you make your changes or implement your new tool.
Once complete, head to  [GitHub to create a pull 
request](https://github.com/pyplati/platipy/compare) to merge your changes into the main branch
(**master**). At this point the automated tests will run and maintainers will review your
submission before merging.

## Example Installation

To install a development environment run the following within a virtual environment:

```bash
pip install -r requirements.txt -r requirements-dev.txt -r requirements-backend.txt
pip install -e .
```

## Style Guide

Python code should conform to
[PEP 8 Style Guide for Python](https://www.python.org/dev/peps/pep-0008/). You may like to use
[*black*](https://github.com/ambv/black) to ensure that your code conforms to PEP 8.

## Structure

In the `platipy` directory you will find the key modules of the library. If you are developing a
new tool/script, create a folder in `platipy/experimental`. We can then collaborate on your tool in
there until we have determined the best place in platipy for your code to live.

## Writing unit tests

Automated unit tests are important for code bases to which various authors are contributing, to
ensure that their changes don't make any unintended breaking changes to other parts of the code.

PlatiPy uses *pytest* as the testing framework. See
[the official pytest documentation](https://docs.pytest.org/en/latest/getting-started.html) for an
introduction to writing tests with pytest.

The automated tests will run when you submit your pull requests. If tests are failing, have a look 
to see what changes could have led to this. Before you code is integrated fully into PlatiPy, you 
should implement some automated tests of your own. If you're unsure how to proceed with this, we
can discuss this in your pull request.

## Providing command line tools

Command line interface (CLI) tools in Platipy use *click*. With *click*. You can find the existing 
CLI tools in `platipy/cli`. Feel free to add a CLI for the tool you are implementing. See 
[the official click documentation](https://click.palletsprojects.com) for more information.
