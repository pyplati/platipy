# Copyright 2022 University of New South Wales, Ingham Institute

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import SimpleITK as sitk

from platipy.backend import app, DataObject, celery  # pylint: disable=unused-import

logger = logging.getLogger(__name__)

SAMPLE_SETTINGS = {
    "HU_BONE_THRESHOLD": 200
}


@app.register("Bone Segmentation Sample", default_settings=SAMPLE_SETTINGS)
def bone_segmentation(data_objects, working_dir, settings):
    """Apply a simple threshold to segment bone on the input CT image

    Args:
        data_objects (list): List of data objects, should contain one fat and one water image
        working_dir (str): Path to directory used for working
        settings ([type]): The settings to use for segmentation

    Returns:
        list: List of output data objects
    """

    logger.info("Running bone segmentation sample")
    logger.info("Using settings: %s", settings)

    output_objects = []
    for data_obj in data_objects:

        # Read the image series
        load_path = data_obj.path
        if data_obj.type == "DICOM":
            load_path = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(data_obj.path)
        img = sitk.ReadImage(load_path)

        # Threshold the CT image
        threshold = settings["HU_BONE_THRESHOLD"]
        mask = img > threshold

        # Create the output Data Objects and add it to output_obj
        mask_file = working_dir.joinpath("mask.nii.gz")
        sitk.WriteImage(mask, mask_file)

        mask_data_object = DataObject(type="FILE", path=mask_file, parent=data_obj)
        output_objects.append(mask_data_object)

    return output_objects


if __name__ == "__main__":

    # Run app by calling "python service.py" from the command line

    DICOM_LISTENER_PORT = 7777
    DICOM_LISTENER_AETITLE = "BONE_SEGMENTATION"

    app.run(
        debug=True,
        host="0.0.0.0",
        port=8000,
        dicom_listener_port=DICOM_LISTENER_PORT,
        dicom_listener_aetitle=DICOM_LISTENER_AETITLE,
    )
