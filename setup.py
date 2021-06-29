import setuptools

import platipy

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fp:
    requirements = list(filter(bool, (line.strip() for line in fp)))

setuptools.setup(
    name=platipy.__name__,
    version=platipy.__version__,
    keywords=platipy.__keywords__,
    author=platipy.__author__,
    author_email=platipy.__author_email__,
    description="Processing Library and Analysis Toolkit for Medical Imaging in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pyplati.github.io/platipy/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Visualization",
        "Development Status :: 4 - Beta",
    ],
    entry_points={
        "console_scripts": [
            "platipy=platipy.cli.run:platipy_cli",
        ]
    },
    license="Apache 2.0 License",
    install_requires=requirements,
)
