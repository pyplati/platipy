# PlatiPy Deployment Service Framework

This framework allows easy deployment of a image analysis tool (such as an auto-segmentation tool)
using a client-server architecture. This lets you run the tool on a server (where you have
sufficient computing resources) and send data to it from a client (which might be a script running
on a desktop PC or a client tool implemented as a plug-in or extension to a commerical system).

> Caution: This framework is designed for use in a Local Area Network (LAN) only. It is not
recommended to use this tool on the WWW without further security considerations being made.

## Prerequisites

This guide assumes you are developing on a Ubuntu Operating System. If you don't have Ubuntu
consider using the [PlatiPy docker image](https://hub.docker.com/r/platipy/platipy)

Make sure you have platipy installed:

```bash
pip install platipy
```

You will also need to install the requirements listed [here](https://github.com/pyplati/platipy/blob/master/requirements-backend.txt).

```bash
wget https://raw.githubusercontent.com/pyplati/platipy/master/requirements-backend.txt
pip install -r requirements-backend.txt
```

Create a directory to develop your service in. In this guide this will be refered to as 
`[working_dir]`. Make sure you are working from within this path:

```bash
cd [working_dir]
```

## Developing a service

### Implementation

Use the following template to implement a service within the framework. See the TODO note where you
can add a call to your function providing some algorithm. Save it in a file named
`[working_dir]/service.py`

```python
import os
import pydicom
import SimpleITK as sitk
import logging
logger = logging.getLogger(__name__)

from platipy.backend import app, DataObject, celery  # pylint: disable=unused-import
from platipy.dicom.io.nifti_to_rtstruct import convert_nifti

# Specify some settings which can be provided by the client calling the algorithm
MY_SETTINGS_DEFAULTS = {
    "outputContourName": "auto_contour_x",
}


@app.register("My Segmentation Tool", default_settings=MY_SETTINGS_DEFAULTS)
def my_segmentation_tool(data_objects, working_dir, settings):

    logger.info("Running My Segmentation Tool")
    logger.info("Using settings: %s", settings)

    output_objects = []
    for d in data_objects:
        logger.info("Running on data object: %s", d.path)

        # Read the image series
        load_path = d.path
        if d.type == "DICOM":
            load_path = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(d.path)

        img = sitk.ReadImage(load_path)

        # TODO Implement your function which returns a segmentation mask here
        mask = my_segmentation_function(img)

        # Write the mask to a file in the working_dir
        mask_file = os.path.join(
            working_dir, "{0}.nii.gz".format(settings["outputContourName"])
        )
        sitk.WriteImage(mask, mask_file)

        # Create the output Data Object and add it to the list of output_objects
        data_object = DataObject(type="FILE", path=mask_file, parent=d)
        output_objects.append(data_object)

        # If the input was a DICOM, then we can use it to generate an output RTStruct
        if d.type == "DICOM":

            dicom_file = load_path[0]
            logger.info("Will write Dicom using file: %s", dicom_file)
            masks = {settings["outputContourName"]: mask_file}

            # Use the image series UID for the file of the RTStruct
            suid = pydicom.dcmread(dicom_file).SeriesInstanceUID
            output_file = os.path.join(working_dir, "RS.{0}.dcm".format(suid))

            # Use the convert nifti function to generate RTStruct from nifti masks
            convert_nifti(dicom_file, masks, output_file)

            # Create the Data Object for the RTStruct and add it to the list
            do = DataObject(type="DICOM", path=output_file, parent=d)
            output_objects.append(do)

            logger.info("RTStruct generated")

    return output_objects


if __name__ == "__main__":

    # Run app by calling "python sample.py" from the command line

    dicom_listener_port = 7777
    dicom_listener_aetitle = "SAMPLE_SERVICE"

    app.run(
        debug=True,
        host="0.0.0.0",
        port=8000,
        dicom_listener_port=dicom_listener_port,
        dicom_listener_aetitle=dicom_listener_aetitle,
    )
```

### Initialise the database

Before running the service, you must initialise the database:

```bash
platipy manage initdb
```

### Add an API Key

And add an API key for use by a client

```bash
platipy manage key --add test
```

Note down the API key for use in the client later on. If you want to list the API keys again you
can use the command:

```bash
platipy manage key --list
```

### Run the service

You can run the service for development and testing purposes using the following command:

```bash
platipy manage run
```

> Note: This way of running the service if for development only. See [Deploying a service](#Deploying-a-service) for
> information on deploying a service in production.
## Developing a client

Client to communicate with your service can be developed in any programming language or environment
you like. Here we give an example using the PlatiPy Client which is a Python solution:

Save the following in a file named `[working_dir]/client.py`. Make sure you replace the API key
generated above in the code below.

```python
import os
import logging
logger = logging.getLogger(__name__)

from platipy.client import PlatiPyClient
from platipy.imaging.tests.data import get_lung_nifti

host = "127.0.0.1" # Set the host name or IP of the server running the service here
port = 8000 # Set the port the service was configured to run on here

api_key = "" # INSERT THE API KEY GENERATED HERE

algorithm_name = "My Segmentation Tool" # The name of the algorithm

# Fetch some data
images = get_lung_nifti()

# Initialise the client
client = PlatiPyClient(host, port, api_key, algorithm_name)

# Add a dataset
dataset = client.add_dataset()

# Create a data object from a CT file
ct_file = list(images.glob("**/*_CT_*.nii.gz"))[0]
data_object = client.add_data_object(dataset, file_path=ct_file)

# Fetch the default settings and override the contour name setting
settings = client.get_default_settings()
settings["outputContourName"] = "CONTOUR_X"

# Run the algorithm and wait for it to finish
for status in client.run_algorithm(dataset, config=settings):
    print('.', end='')

# Download the output data object and save it into the current directory
client.download_output_objects(dataset, output_path=".")
```

Run the client script with:

```bash
python client.py
```

## Deploying a service

> Coming soon

### Build a docker image

### Secure using SSL

