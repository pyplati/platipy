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

import tempfile

from loguru import logger

from platipy.imaging.projects.bronchus.bronchus import (
    generate_lung_mask,
    generate_airway_mask,
    default_settings
)

BRONCHUS_SETTINGS_DEFAULTS = {
    "outputBronchusName": "Auto_Bronchus",
    "outputLungName": "Auto_Lung",
    "algorithmSettings": default_settings,
}


def run_bronchus_segmentation(input_image, settings=BRONCHUS_SETTINGS_DEFAULTS):
    """Runs the Proximal Bronchial Tree segmentation

    Args:
        input_image (sitk.Image): SimpleITK image on which to perform the segmentation
        settings (dict, optional): Dictionary containing settings for algorithm.
                                   Defaults to BRONCHUS_SETTINGS_DEFAULTS.

    Returns:
        dict: Dictionary containing output of segmentation
    """

    working_directory = tempfile.mkdtemp()
    results = {}

    # Compute the lung mask
    lung_mask = generate_lung_mask(input_image)
    results[settings["outputLungName"]] = lung_mask

    bronchus_mask = generate_airway_mask(
        working_directory,
        input_image,
        lung_mask,
        config_dict=settings["algorithmSettings"],
    )

    # If the bronchus mask counldn't be generated then skip it
    if not bronchus_mask:
        logger.error("Unable to generate bronchus mask")
        return results

    results[settings["outputBronchusName"]] = bronchus_mask

    return results
