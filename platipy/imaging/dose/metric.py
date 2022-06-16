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

import numpy as np
import SimpleITK as sitk


def calculate_d_mean(dose_grid, label):
    """Calculate the mean dose of a structure

    Args:
        dose_grid (SimpleITK.Image): The dose grid.
        label (SimpleITK.Image): The (binary) label defining a structure.

    Returns:
        float: The mean dose in Gy.
    """

    dose_grid = sitk.Resample(dose_grid, label, sitk.Transform(), sitk.sitkLinear)
    dose_array = sitk.GetArrayFromImage(dose_grid)
    mask_array = sitk.GetArrayFromImage(label)

    return dose_array[mask_array > 0].mean()


def calculate_d_max(dose_grid, label):
    """Calculate the maximum dose of a structure

    Args:
        dose_grid (SimpleITK.Image): The dose grid.
        label (SimpleITK.Image): The (binary) label defining a structure.

    Returns:
        float: The maximum dose in Gy.
    """

    dose_grid = sitk.Resample(dose_grid, label, sitk.Transform(), sitk.sitkLinear)
    dose_array = sitk.GetArrayFromImage(dose_grid)
    mask_array = sitk.GetArrayFromImage(label)

    return dose_array[mask_array > 0].max()


def calculate_d_to_volume(dose_grid, label, volume, volume_in_cc=False):
    """Calculate the dose to a (relative) volume of the label

    Args:
        dose_grid (SimpleITK.Image): The dose grid.
        label (SimpleITK.Image): The (binary) label defining a structure.
        volume (float): The relative volume in %.
        volume_in_cc (bool, optional): Whether the volume is in cc (versus percent).
            Defaults to False.

    Returns:
        float: The dose to volume ratio.
    """

    dose_grid = sitk.Resample(dose_grid, label, sitk.Transform(), sitk.sitkLinear)
    dose_array = sitk.GetArrayFromImage(dose_grid)
    mask_array = sitk.GetArrayFromImage(label)

    if volume_in_cc:
        volume = volume * 1000 / ((mask_array > 0).sum() * np.product(label.GetSpacing()))

    return np.percentile(dose_array[mask_array > 0], volume)


def calculate_v_receiving_dose(dose_grid, label, dose_threshold=50):
    """Calculate the (relative) volume receiving a dose above a threshold

    Args:
        dose_grid (SimpleITK.Image): The dose grid.
        label (SimpleITK.Image): The (binary) label defining a structure.
        dose_threshold (float, optional): The dose threshold in Gy. Defaults to 50.

    Returns:
        float: The (relative) volume receiving a dose above the threshold, as a percent.
    """

    dose_grid = sitk.Resample(dose_grid, label, sitk.Transform(), sitk.sitkLinear)
    dose_array = sitk.GetArrayFromImage(dose_grid)
    mask_array = sitk.GetArrayFromImage(label)

    dose_array_masked = dose_array[mask_array > 0]

    num_voxels = (mask_array > 0).sum()

    return (dose_array_masked >= dose_threshold).sum() / num_voxels * 100