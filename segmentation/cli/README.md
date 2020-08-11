# PlatiPy Segmentation Command Line Interface

## Description

This command line tool provides a simple interface to run the segmentation tools available within
PlatiPy.

## Usage

### 1. Add library to python path

Ensure that the repository path has been added to your Python environment. You can use the
following command:

```
export PYTHONPATH=/path/to/platipy:$PYTHONPATH
```

### 2. Run the CLI

You can then run the CLI with to print the help:

```
python run.py --help
```

Select the segmentation algorithm to run (cardiac for this example), and see what default settings
it uses:

```
python run.py cardiac -d
```

You probably want to make changes to these default settings to run it on your data. You can
redirect these default settings to a file so that you can make your changes:

```
python run.py cardiac -d > cardiac_settings.json
```

Make the required changes to the algorithm settings in the json file. Then run the segmentation on
a new case using:

```
python run.py cardiac -c cardiac_settings.json path/to/case.nii.gz
```

Note: The command line interface will accept an image in Nifti format or a directory containing a
Dicom series as input.
