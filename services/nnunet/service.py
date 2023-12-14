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
import json

from pathlib import Path

import logging
import SimpleITK as sitk

from platipy.backend import app, DataObject, celery  # pylint: disable=unused-import

logger = logging.getLogger(__name__)

NNUNET_SETTINGS_DEFAULTS = {
    "task": "TaskXXX",
    "config": "2d",
    "trainer": None,
    "fold": None,
    "clean_sup_slices": False,
}


def clean_sup_slices(mask):
    lssif = sitk.LabelShapeStatisticsImageFilter()
    max_slice_size = 0
    sizes = {}
    for z in range(mask.GetSize()[2] - 1, -1, -1):
        lssif.Execute(sitk.ConnectedComponent(mask[:, :, z]))
        if len(lssif.GetLabels()) == 0:
            continue

        phys_size = lssif.GetPhysicalSize(1)

        if phys_size > max_slice_size:
            max_slice_size = phys_size

        sizes[z] = phys_size
    for z in sizes:
        if sizes[z] > max_slice_size / 2:
            mask[:, :, z + 1 : mask.GetSize()[2]] = 0
            break

    return mask


def get_structure_names(task):
    # Look up structure names if we can find them dataset.json file
    if "nnUNet_raw_data_base" not in os.environ:
        logger.info("nnUNet_raw_data_base not set")
        return {}

    raw_path = Path(os.environ["nnUNet_raw_data_base"])
    task_path = raw_path.joinpath("nnUNet_raw_data", task)
    dataset_file = task_path.joinpath("dataset.json")

    logger.info("Attempting to read %s", dataset_file)

    if not dataset_file.exists():
        logger.info("dataset.json file does not exist for %s", dataset_file)
        return {}

    dataset = {}
    with open(dataset_file, "r") as f:
        dataset = json.load(f)

    if "labels" not in dataset:
        logger.info("Something went wrong reading dataset.json file")
        return {}

    return dataset["labels"]


@app.register("nnUNet Service", default_settings=NNUNET_SETTINGS_DEFAULTS)
def nnunet_service(data_objects, working_dir, settings):
    """
    Run a nnUNet task
    """

    output_objects = []

    logger.info("Running nnUNet")
    logger.info("Using settings: %s", settings)
    logger.info("Working Dir: %s", working_dir)

    input_path = Path(working_dir).joinpath("input")
    input_path.mkdir()

    output_path = Path(working_dir).joinpath("output")
    output_path.mkdir()

    labels = get_structure_names(settings["task"])
    logger.info("Read labels: %s", labels)

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

        if "fold" in settings and settings["fold"]:
            command += ["-f", settings["fold"]]

        if "trainer" in settings and settings["trainer"]:
            command += ["-tr", settings["trainer"]]

        logger.info("Running command: %s", command)
        subprocess.call(command)

        for op in output_path.glob("*.nii.gz"):
            label_map = sitk.ReadImage(str(op))

            label_map_arr = sitk.GetArrayFromImage(label_map)
            label_count = label_map_arr.max()

            for label_id in range(1, label_count + 1):
                mask = label_map == label_id

                label_name = f"Structure_{label_id}"
                if str(label_id) in labels:
                    label_name = labels[str(label_id)]

                if settings["clean_sup_slices"]:
                    mask = clean_sup_slices(mask)

                mask_file = output_path.joinpath(f"{label_name}.nii.gz")

                sitk.WriteImage(mask, str(mask_file))

            output_data_object = DataObject(
                type="FILE", path=str(mask_file), parent=data_object
            )
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
