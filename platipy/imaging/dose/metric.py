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
import pandas as pd


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
        volume = (volume * 1000 / ((mask_array > 0).sum() * np.product(label.GetSpacing()))) * 100

    if volume > 100:
        volume = 100

    return np.percentile(dose_array[mask_array > 0], 100 - volume)


def calculate_v_receiving_dose(dose_grid, label, dose_threshold, relative=True):
    """Calculate the (relative) volume receiving a dose above a threshold

    Args:
        dose_grid (SimpleITK.Image): The dose grid.
        label (SimpleITK.Image): The (binary) label defining a structure.
        dose_threshold (float): The dose threshold in Gy.
        relative (bool, optional): If true results will be returned as relative volume, otherwise
            as volume in cc. Defaults to True.

    Returns:
        float: The (relative) volume receiving a dose above the threshold, as a percent.
    """

    dose_grid = sitk.Resample(dose_grid, label, sitk.Transform(), sitk.sitkLinear)
    dose_array = sitk.GetArrayFromImage(dose_grid)
    mask_array = sitk.GetArrayFromImage(label)

    dose_array_masked = dose_array[mask_array > 0]

    num_voxels = (mask_array > 0).sum()

    relative_volume = (dose_array_masked >= dose_threshold).sum() / num_voxels * 100
    if relative:
        return relative_volume

    total_volume = (mask_array > 0).sum() * np.product(label.GetSpacing()) / 1000

    return relative_volume * total_volume


def calculate_d_to_volume_for_labels(dose_grid, labels, volume, volume_in_cc=False):
    """Calculate the dose which x percent of the volume receives for a set of labels

    Args:
        dose_grid (SimpleITK.Image): The dose grid.
        labels (dict): A Python dictionary containing the label name as key and the SimpleITK.Image
          binary mask as value.
        volume (float|list): The relative volume (or list of volumes) in %.
        volume_in_cc (bool, optional): Whether the volume is in cc (versus percent).
            Defaults to False.

    Returns:
        pandas.DataFrame: Data frame with a row for each label containing the metric and value.
    """

    if not isinstance(volume, list):
        volume = [volume]

    metrics = []
    for label in labels:

        m = {"label": label}

        for v in volume:
            col_name = f"D{v}"
            if volume_in_cc:
                col_name = f"D{v}cc"

            m[col_name] = calculate_d_to_volume(
                dose_grid, labels[label], v, volume_in_cc=volume_in_cc
            )

        metrics.append(m)

    return pd.DataFrame(metrics)


def calculate_v_receiving_dose_for_labels(dose_grid, labels, dose_threshold, relative=True):
    """Get the volume (in cc) which receives x dose for a set of labels

    Args:
        dose_grid (SimpleITK.Image): The dose grid.
        labels (SimpleITK.Image): The (binary) label defining a structure.
        dose_threshold (float|list): The dose threshold (or list of thresholds) in Gy.
        relative (bool, optional): If true results will be returned as relative volume, otherwise
            as volume in cc. Defaults to True.

    Returns:
        pandas.DataFrame: Data frame with a row for each label containing the metric and value.
    """

    if not isinstance(dose_threshold, list):
        dose_threshold = [dose_threshold]

    metrics = []
    for label in labels:

        m = {"label": label}

        for dt in dose_threshold:

            metric_name = f"V{dt}"
            if dt - int(dt) == 0:
                metric_name = f"V{int(dt)}"

            m[metric_name] = calculate_v_receiving_dose(dose_grid, labels[label], dt, relative)

        metrics.append(m)

    return pd.DataFrame(metrics)
