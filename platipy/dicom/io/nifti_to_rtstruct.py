# Copyright 2020 University of New South Wales, University of Sydney

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script was provided by Jason Dowling and written by Parnesh Raniga, CSIRO, Brisbane.


from pathlib import Path
import SimpleITK as sitk
import numpy as np
from loguru import logger
from matplotlib import cm

from rt_utils import RTStructBuilder


def convert_nifti(dcm_path, mask_input, output_file, color_map=cm.get_cmap("rainbow")):
    """Convert a set of masks to a DICOM RTStruct object.

    This function now utilises the rt-utils package: https://github.com/qurit/rt-utils
    We keep this function in here for compatibility

    Args:
        dcm_path (str|pathlib.Path): Path to the reference DICOM series
        mask_input (dict|list): A dictionary containing the name as key and image as value. Or a
                                list of string with comma separated name and mask paths (name,path)
        output_file (str|pathlib.Path): The path to the file to write the RTStruct
         color_map (matplotlib.colors.Colormap, optional): Colormap to use for output. Defaults to
                                                           cm.get_cmap("rainbow").
    """

    logger.info("Will convert the following Nifti masks to RTStruct:")

    masks = {}  # Dict stores key value pairs for masks
    if isinstance(mask_input, dict):
        # Dict was already passed in
        masks = mask_input
    else:
        # Otherwise convert list of comma separated masks
        for mask in mask_input:
            mask_parts = mask.split(",")
            masks[mask_parts[0]] = mask_parts[1]

    if not isinstance(dcm_path, Path):
        dcm_path = Path(dcm_path)

    dcm_series_path = None
    if dcm_path.is_file():
        dcm_series_path = dcm_path.parent
    else:
        dcm_series_path = dcm_path

    rtstruct = RTStructBuilder.create_new(dicom_series_path=str(dcm_series_path))

    for mask_name in masks:

        # Use a hash of the name to get the color from the supplied color map
        color = color_map(hash(mask_name) % 256)
        color = color[:3]
        color = [int(c * 255) for c in color]

        mask = masks[mask_name]
        if not isinstance(mask, sitk.Image):
            mask = sitk.ReadImage(str(mask))

        bool_arr = sitk.GetArrayFromImage(mask) != 0
        bool_arr = np.transpose(bool_arr, (1, 2, 0))
        rtstruct.add_roi(mask=bool_arr, color=color, name=mask_name)

    rtstruct.save(str(output_file))
