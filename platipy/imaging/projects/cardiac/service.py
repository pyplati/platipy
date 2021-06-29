# Copyright 2020 University of New South Wales, University of Sydney, Ingham Institute

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os

import SimpleITK as sitk
from loguru import logger

# import pydicom

# Need include celery here to be able to from Docker container
# pylint: disable=unused-import
from platipy.backend import app, DataObject, celery

from platipy.imaging.projects.cardiac.run import (
    CARDIAC_SETTINGS_DEFAULTS,
    run_cardiac_segmentation,
)


@app.register("Cardiac Segmentation", default_settings=CARDIAC_SETTINGS_DEFAULTS)
def cardiac_service(data_objects, working_dir, settings):
    """
    Implements the platipy framework to provide cardiac atlas based segmentation.
    """

    logger.info("Running Cardiac Segmentation")
    logger.info("Using settings: " + str(settings))

    output_objects = []
    for data_object in data_objects:
        logger.info("Running on data object: " + data_object.path)

        # Read the image series
        load_path = data_object.path
        if data_object.type == "DICOM":
            load_path = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(data_object.path)

        img = sitk.ReadImage(load_path)

        results, _ = run_cardiac_segmentation(img, settings=settings)

        # Save resulting masks and add to output for service
        for output in results:

            mask_file = os.path.join(working_dir, "{0}.nii.gz".format(output))
            sitk.WriteImage(results[output], mask_file)

            output_data_object = DataObject(type="FILE", path=mask_file, parent=data_object)
            output_objects.append(output_data_object)

        # If the input was a DICOM, then we can use it to generate an output RTStruct
        # if data_object.type == "DICOM":

        #     dicom_file = load_path[0]
        #     logger.info("Will write Dicom using file: {0}".format(dicom_file))
        #     masks = {settings["outputContourName"]: mask_file}

        #     # Use the image series UID for the file of the RTStruct
        #     suid = pydicom.dcmread(dicom_file).SeriesInstanceUID
        #     output_file = os.path.join(working_dir, "RS.{0}.dcm".format(suid))

        #     # Use the convert nifti function to generate RTStruct from nifti masks
        #     convert_nifti(dicom_file, masks, output_file)

        #     # Create the Data Object for the RTStruct and add it to the list
        #     do = DataObject(type="DICOM", path=output_file, parent=d)
        #     output_objects.append(do)

        #     logger.info("RTStruct generated")

    return output_objects


@app.register(
    "Cardiac Structure Guided Segmentation",
    default_settings=CARDIAC_SETTINGS_DEFAULTS,
)
def cardiac_structure_guided_service(data_objects, working_dir, settings):
    """Runs the structure guided cardiac segmentation service"""

    logger.info("Running Structure Guided Cardiac Segmentation")
    logger.info("Using settings: " + str(settings))

    output_objects = []
    for data_object in data_objects:
        logger.info("Running on data object: " + data_object.path)

        # Read the image series
        load_path = data_object.path
        if data_object.type == "DICOM":
            load_path = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(data_object.path)

        img = sitk.ReadImage(load_path)

        # Load the WHOLEHEART contour (child of Image)
        if len(data_object.children) == 0:
            logger.error(
                "Wholeheart structure needed for structure guided cardiac "
                f"segmentation, skipping {data_object.id}"
            )
            continue

        wholeheart = sitk.ReadImage(data_object.children[0].path)

        results = run_cardiac_segmentation(img, wholeheart, settings)

        # Save resulting masks and add to output for service
        for output in results:

            mask_file = os.path.join(working_dir, "{0}.nii.gz".format(output))
            sitk.WriteImage(results[output], mask_file)

            output_data_object = DataObject(type="FILE", path=mask_file, parent=data_object)
            output_objects.append(output_data_object)

    return output_objects


if __name__ == "__main__":

    # Run app by calling "python sample.py" from the command line

    DICOM_LISTENER_PORT = 7777
    DICOM_LISTENER_AETITLE = "SAMPLE_SERVICE"

    app.run(
        debug=True,
        host="0.0.0.0",
        port=8000,
        dicom_listener_port=DICOM_LISTENER_PORT,
        dicom_listener_aetitle=DICOM_LISTENER_AETITLE,
    )
