# Copyright 2020 CSIRO, University of New South Wales, Ingham Institute

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

# import pydicom

import SimpleITK as sitk
from loguru import logger

from platipy.backend import app, DataObject, celery

from platipy.imaging.projects.bronchus.run import (
    run_bronchus_segmentation,
    BRONCHUS_SETTINGS_DEFAULTS
)


@app.register("Bronchus Segmentation", default_settings=BRONCHUS_SETTINGS_DEFAULTS)
def bronchus_service(data_objects, working_dir, settings):
    """
    Implements the platipy framework to provide bronchus segmentation.
    """

    logger.info("Running Bronchus Segmentation")

    output_objects = []
    for data_object in data_objects:
        logger.info("Running on data object: " + data_object.path)

        # Read the image series
        load_path = data_object.path
        if data_object.type == "DICOM":
            load_path = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(
                data_object.path
            )

        img = sitk.ReadImage(load_path)
        results = run_bronchus_segmentation(img, settings)

        # Save resulting masks and add to output for service
        for output in results.keys():

            mask_file = os.path.join(working_dir, "{0}.nii.gz".format(output))
            sitk.WriteImage(results[output], mask_file)

            output_data_object = DataObject(
                type="FILE", path=mask_file, parent=data_object
            )
            output_objects.append(output_data_object)

        # If the input was a DICOM, then we can use it to generate an output RTStruct
        # if d.type == 'DICOM':

        #     dicom_file = load_path[0]
        #     logger.info('Will write Dicom using file: {0}'.format(dicom_file))
        #     masks = {settings['outputContourName']: mask_file}

        #     # Use the image series UID for the file of the RTStruct
        #     suid = pydicom.dcmread(dicom_file).SeriesInstanceUID
        #     output_file = os.path.join(working_dir, 'RS.{0}.dcm'.format(suid))

        #     # Use the convert nifti function to generate RTStruct from nifti masks
        #     convert_nifti(dicom_file, masks, output_file)

        #     # Create the Data Object for the RTStruct and add it to the list
        #     do = DataObject(type='DICOM', path=output_file, parent=d)
        #     output_objects.append(do)

        #     logger.info('RTStruct generated')

    return output_objects


if __name__ == "__main__":

    # Run app by calling "python bronchus.py" from the command line

    DICOM_LISTENER_PORT = 7777
    DICOM_LISTENER_AETITLE = "BRONCHUS_SERVICE"

    app.run(
        debug=True,
        host="0.0.0.0",
        port=8000,
        dicom_listener_port=DICOM_LISTENER_PORT,
        dicom_listener_aetitle=DICOM_LISTENER_AETITLE,
    )
