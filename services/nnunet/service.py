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

from loguru import logger
import SimpleITK as sitk

from platipy.backend import app, DataObject, celery


NNUNET_SETTINGS_DEFAULTS = {
    "task": "TaskXXX",
    "config": "2d",
    "trainer": None,
    "clean_sup_slices": False,
}

def clean_sup_slices(mask):
    lssif = sitk.LabelShapeStatisticsImageFilter()
    max_slice_size = 0
    sizes = {}
    for z in range(mask.GetSize()[2]-1, -1, -1):
        lssif.Execute(sitk.ConnectedComponent(mask[:,:,z]))
        if len(lssif.GetLabels()) == 0:
            continue

        phys_size = lssif.GetPhysicalSize(1)

        if phys_size > max_slice_size:
            max_slice_size = phys_size

        sizes[z] = phys_size
    for z in sizes:
        if sizes[z] > max_slice_size/2:
            mask[:,:,z+1:mask.GetSize()[2]] = 0
            break

    return mask


@app.register("nnUNet Service", default_settings=NNUNET_SETTINGS_DEFAULTS)
def nnunet_service(data_objects, working_dir, settings):
    """
    Run a nnUNet task
    """

    output_objects = []

    logger.info("Running nnUNet")
    logger.info("Using settings: {0}".format(settings))
    logger.info("Working Dir: {0}".format(working_dir))

    input_path = Path(working_dir).joinpath("input")
    input_path.mkdir()

    output_path = Path(working_dir).joinpath("output")
    output_path.mkdir()

    for data_object in data_objects:

        # Create a symbolic link for each image to auto-segment using the nnUNet
        do_path = Path(data_object.path)
        io_path = input_path.joinpath(f"{settings['task']}_0000.nii.gz")
        load_path = data_object.path
        if data_object.type == "DICOM":
            load_path = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(
                data_object.path
            )

        img = sitk.ReadImage(load_path)
        sitk.WriteImage(img, str(io_path))

        command = [
            "nnUNet_predict",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "-t",
            settings["task"],
            "-m",
            settings["config"],
        ]

        if settings["trainer"]:
            command += ["-tr", settings["trainer"]]

        logger.info(f"Running command: {command}")
        subprocess.call(command)

        for op in output_path.glob("*.nii.gz"):

            if settings["clean_sup_slices"]:
                mask = sitk.ReadImage(str(op))
                mask = clean_sup_slices(mask)
                sitk.WriteImage(mask, str(op))

            output_data_object = DataObject(type="FILE", path=str(op), parent=data_object)
            output_objects.append(output_data_object)

        os.remove(io_path)

    logger.info("Finished running nnUNet")

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
