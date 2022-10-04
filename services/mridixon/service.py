# Copyright 2020 University of New South Wales, Ingham Institute

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

import logging
logger = logging.getLogger(__name__)

import SimpleITK as sitk

from platipy.backend import app, DataObject, celery  # pylint: disable=unused-import

MRI_DIXON_SETTINGS = {}


@app.register("MRI Dixon Analysis", default_settings=MRI_DIXON_SETTINGS)
def mri_dixon_analysis(data_objects, working_dir, settings):
    """Calculate Fat Water fraction for appropriate MRI Dixon images

    Args:
        data_objects (list): List of data objects, should contain one fat and one water image
        working_dir (str): Path to directory used for working
        settings ([type]): The settings to use for analysis

    Returns:
        list: List of output data objects
    """

    logger.info("Running Dixon analysis Calculation")
    logger.info("Using settings: %s", settings)

    output_objects = []

    fat_obj = None
    water_obj = None
    for data_obj in data_objects:

        if data_obj.meta_data["image_type"] == "fat":
            fat_obj = data_obj

        if data_obj.meta_data["image_type"] == "water":
            water_obj = data_obj

    if fat_obj is None or water_obj is None:
        logger.error("Both Fat and Water Images are required")
        return []

    # Read the image series
    fat_load_path = fat_obj.path
    if fat_obj.type == "DICOM":
        fat_load_path = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(fat_obj.path)
    fat_img = sitk.ReadImage(fat_load_path)

    water_load_path = water_obj.path
    if water_obj.type == "DICOM":
        water_load_path = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(water_obj.path)
    water_img = sitk.ReadImage(water_load_path)

    # Cast to float for calculation
    fat_img = sitk.Cast(fat_img, sitk.sitkFloat32)
    water_img = sitk.Cast(water_img, sitk.sitkFloat32)

    # Let's do the calcuation using NumPy
    fat_arr = sitk.GetArrayFromImage(fat_img)
    water_arr = sitk.GetArrayFromImage(water_img)

    # Do the calculation
    divisor = water_arr + fat_arr
    fat_fraction_arr = (fat_arr * 100) / divisor
    fat_fraction_arr[divisor == 0] = 0  # Sets those voxels which were divided by zero to 0
    water_fraction_arr = (water_arr * 100) / divisor
    water_fraction_arr[divisor == 0] = 0  # Sets those voxels which were divided by zero to 0

    fat_fraction_img = sitk.GetImageFromArray(fat_fraction_arr)
    water_fraction_img = sitk.GetImageFromArray(water_fraction_arr)

    fat_fraction_img.CopyInformation(fat_img)
    water_fraction_img.CopyInformation(water_img)

    # Create the output Data Objects and add it to output_ob
    fat_fraction_file = os.path.join(working_dir, "fat.nii.gz")
    sitk.WriteImage(fat_fraction_img, fat_fraction_file)
    water_fraction_file = os.path.join(working_dir, "water.nii.gz")
    sitk.WriteImage(water_fraction_img, water_fraction_file)

    fat_data_object = DataObject(type="FILE", path=fat_fraction_file, parent=fat_obj)
    output_objects.append(fat_data_object)

    water_data_object = DataObject(type="FILE", path=water_fraction_file, parent=water_obj)
    output_objects.append(water_data_object)

    return output_objects


if __name__ == "__main__":

    # Run app by calling "python service.py" from the command line

    DICOM_LISTENER_PORT = 7777
    DICOM_LISTENER_AETITLE = "MRI_DIXON_ANALYSIS"

    app.run(
        debug=True,
        host="0.0.0.0",
        port=8000,
        dicom_listener_port=DICOM_LISTENER_PORT,
        dicom_listener_aetitle=DICOM_LISTENER_AETITLE,
    )
