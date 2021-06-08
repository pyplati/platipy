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


def calculate_dvh(dose_grid, label, bins=1001):
    """Calculates a dose-volume histogram

    Args:
        dose_grid (SimpleITK.Image): The dose grid.
        label (SimpleITK.Image): The (binary) label defining a structure.
        bins (int | list | np.ndarray, optional): Passed to np.histogram,
            can be an int (number of bins), or a list (specifying bin edges). Defaults to 1001.

    Returns:
        SimpleITK.Image: [description]
    """

    if dose_grid.GetSize() != label.GetSize():
        print("Dose grid size does not match label, automatically resampling.")
        dose_grid = sitk.Resample(dose_grid, label)

    dose_arr = sitk.GetArrayViewFromImage(dose_grid)
    label_arr = sitk.GetArrayViewFromImage(label)

    dose_vals = dose_arr[np.where(label_arr)]

    counts, bin_edges = np.histogram(dose_vals, bins=bins)

    # Get mid-points of bins
    dose_points = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    # Calculate the actual DVH values
    counts = np.cumsum(counts[::-1])[::-1]
    counts = counts / counts.max()

    return dose_points, counts
