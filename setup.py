import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fp:
    requirements = list(filter(bool, (line.strip() for line in fp)))

setuptools.setup(
    name="platipy",
    version="0.1.0",
    author="Phillip Chlap & Robert Finnegan",
    author_email="phillip.chlap@unsw.edu.au",
    description="Processing Library and Analysis Toolkit for Medical Imaging in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pyplati/platipy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Visualization",
        "Development Status :: 3 - Alpha",
    ],
    entry_points={
        "console_scripts": [
            "platipy=platipy.cli.run:platipy_cli",
        ]
    },
    license="Apache 2.0 License",
    install_requires=requirements,
)
