# Copyright 2021 University of New South Wales, University of Sydney, Ingham Institute

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
import subprocess

from pathlib import Path

import logging
import SimpleITK as sitk

from totalsegmentator.python_api import totalsegmentator

from platipy.backend import app, DataObject, celery  # pylint: disable=unused-import

logger = logging.getLogger(__name__)

TOTALSEG_SETTINGS_DEFAULTS = {
    "fast": False,
    "body_seg": False,
}


@app.register("TotalSegmentator Service", default_settings=TOTALSEG_SETTINGS_DEFAULTS)
def totalsegmentator_service(data_objects, working_dir, settings):
    """
    Run the TotalSegmentator
    """

    output_objects = []

    logger.info("Running TotalSegmentator")
    logger.info("Using settings: %s", settings)
    logger.info("Working Dir: %s", working_dir)

    input_path = Path(working_dir).joinpath("input")
    input_path.mkdir()

    output_path = Path(working_dir).joinpath("output")
    output_path.mkdir()

    for data_object in data_objects:
        # Create a symbolic link for each image to auto-segment using the nnUNet
        do_path = Path(data_object.path)
        io_path = input_path.joinpath("image_0000.nii.gz")
        load_path = data_object.path
        if data_object.type == "DICOM":
            load_path = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(do_path)

        img = sitk.ReadImage(load_path)
        sitk.WriteImage(img, str(io_path))

        logger.info("Running TotalSegmentator on input path: %s", input_path)
        totalsegmentator(
            input_path,
            output_path,
            fast=default_settings["fast"],
            body_seg=default_settings["body_seg"],
        )

        for op in output_path.glob("*.nii.gz"):
            output_data_object = DataObject(type="FILE", path=str(op), parent=data_object)
            output_objects.append(output_data_object)

        os.remove(io_path)

    logger.info("Finished running TotalSegmentator")

    return output_objects


if __name__ == "__main__":
    # Run app by calling "python service.py" from the command line

    DICOM_LISTENER_PORT = 7777
    DICOM_LISTENER_AETITLE = "NNUNET_EXPORT_SERVICE"

    app.run(
        debug=True,
        host="0.0.0.0",
        port=8001,
        dicom_listener_port=DICOM_LISTENER_PORT,
        dicom_listener_aetitle=DICOM_LISTENER_AETITLE,
    )
