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


def calculate_dvh(dose_grid, label, bins=1001):
    """Calculates a dose-volume histogram

    Args:
        dose_grid (SimpleITK.Image): The dose grid.
        label (SimpleITK.Image): The (binary) label defining a structure.
        bins (int | list | np.ndarray, optional): Passed to np.histogram,
            can be an int (number of bins), or a list (specifying bin edges). Defaults to 1001.

    Returns:
        bins (numpy.ndarray): The points of the dose bins
        values (numpy.ndarray): The DVH values
    """

    if dose_grid.GetSize() != label.GetSize():
        print("Dose grid size does not match label, automatically resampling.")
        dose_grid = sitk.Resample(dose_grid, label)

    dose_arr = sitk.GetArrayViewFromImage(dose_grid)
    label_arr = sitk.GetArrayViewFromImage(label)

    dose_vals = dose_arr[np.where(label_arr)]

    counts, bin_edges = np.histogram(dose_vals, bins=bins)

    # Get mid-points of bins
    bins = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    # Calculate the actual DVH values
    values = np.cumsum(counts[::-1])[::-1]
    if np.all(values == 0):
        return bins, values

    values = values / values.max()

    return bins, values


def calculate_dvh_for_labels(dose_grid, labels, bin_width=0.1, max_dose=None):
    """Calculate the DVH for multiple labels

    Args:
        dose_grid (SimpleITK.Image): Dose grid
        labels (dict): Dictionary of labels with the label name as key and SimpleITK.Image mask as
            value.
        bin_width (float, optional): The width of each bin of the DVH (Gy). Defaults to 0.1.
        max_dose (float, optional): The maximum dose of the DVH. If not set then maximum dose from
            dose grid is used.Defaults to None.

    Returns:
        pandas.DataFrame: The DVH for each structure along with the mean dose and size in cubic
            centimetres as a data frame.
    """

    dvh = []

    label_keys = labels.keys()

    dose_grid = sitk.Resample(dose_grid, labels[list(label_keys)[0]])
    dose_array = sitk.GetArrayFromImage(dose_grid)

    if not max_dose:
        max_dose = dose_array.max()

    for k in label_keys:

        mask = labels[k]
        mask_array = sitk.GetArrayFromImage(mask)

        # Compute cubic centimetre volume of structure
        cc = mask_array.sum() * np.prod([a / 10 for a in mask.GetSpacing()])

        bins, values = calculate_dvh(
            dose_grid, labels[k], bins=np.arange(-bin_width / 2, max_dose + bin_width, bin_width)
        )

        # Remove rounding error
        bins = np.round(
            bins.astype(float),
            decimals=10,
        )

        mean_dose = dose_array[mask_array > 0].mean()
        entry = {
            **{
                "label": k,
                "cc": cc,
                "mean": mean_dose,
            },
            **dict(zip(bins, values)),
        }

        dvh.append(entry)

    return pd.DataFrame(dvh)


def calculate_d_x(dvh, x, label=None):
    """Calculate the dose which x percent of the volume receives

    Args:
        dvh (pandas.DataFrame): DVH DataFrame as produced by calculate_dvh_for_labels
        x (float|list): The dose threshold (or list of dose thresholds) which x percent of the
            volume receives
        label (str, optional): The label to compute the metric for. Computes for all metrics if not
            set. Defaults to None.

    Returns:
        pandas.DataFrame: Data frame with a row for each label containing the metric and value.
    """

    if label:
        dvh = dvh[dvh.label == label]

    if not isinstance(x, list):
        x = [x]

    bins = np.array([b for b in dvh.columns if isinstance(b, float)])
    values = np.array(dvh[bins])

    metrics = []
    for idx in range(len(dvh)):
        d = dvh.iloc[idx]

        m = {"label": d.label}

        for threshold in x:
            value = np.interp(threshold / 100, values[idx][::-1], bins[::-1])
            if values[idx, 0] == np.sum(values[idx]):
                value = 0

            # Interp will return zero when computing D100, do compute this separately
            if threshold == 100:
                i, j = np.where(values == 1.0)
                value = bins[j][i == idx][-1]

            m[f"D{threshold}"] = value

        metrics.append(m)

    return pd.DataFrame(metrics)


def calculate_v_x(dvh, x, label=None):
    """Get the volume (in cc) which receives x dose

    Args:
        dvh (pandas.DataFrame): DVH DataFrame as produced by calculate_dvh_for_labels
        x (float|list): The dose threshold (or list of dose thresholds) to get the volume for.
        label (str, optional): The label to compute the metric for. Computes for all metrics if not
            set. Defaults to None.

    Returns:
        pandas.DataFrame: Data frame with a row for each label containing the metric and value.
    """

    if label:
        dvh = dvh[dvh.label == label]

    if not isinstance(x, list):
        x = [x]

    bins = np.array([b for b in dvh.columns if isinstance(b, float)])
    values = np.array(dvh[bins])

    metrics = []
    for idx in range(len(dvh)):
        d = dvh.iloc[idx]

        m = {"label": d.label}

        for threshold in x:
            value = np.interp(threshold, bins, values[idx]) * d.cc

            metric_name = f"V{threshold}"
            if threshold - int(threshold) == 0:
                metric_name = f"V{int(threshold)}"

            m[metric_name] = value

        metrics.append(m)

    return pd.DataFrame(metrics)


def calculate_d_cc_x(dvh, x, label=None, index_cols=None):
    """Compute the dose which is received by cc of the volume

    Args:
        dvh (pandas.DataFrame): DVH DataFrame as produced by calculate_dvh_for_labels
        x (float|list): The cc (or list of cc's) to compute the dose at.
        label (str, optional): The label to compute the metric for. Computes for all metrics if not
            set. Defaults to None.
        index_cols (list, optional): List of columns to group by when computing the metric.
            Defaults to ["label"].

    Returns:
        pandas.DataFrame: Data frame with a row for each label containing the metric and value.
    """

    # Default to using only label as index_cols
    if index_cols is None:
        index_cols = ["label"]

    if label:
        dvh = dvh[dvh.label == label]

    if not isinstance(x, list):
        x = [x]

    metrics = []
    # Group by struct_hash, dose_hash, and label to ensure unique values per structure
    for idx in dvh.groupby(index_cols).groups.keys():

        if isinstance(idx, str):
            idx = [idx]

        m = {}
        group = dvh
        for i, col in enumerate(index_cols):
            m[col] = idx[i]

            group = group[group[col] == idx[i]]

        for threshold in x:
            # Calculate the dose at the specified cc threshold
            cc_at = (threshold / group.cc.iloc[0]) * 100
            cc_at = min(cc_at, 100)
            cc_val = calculate_d_x(group, cc_at)[f"D{cc_at}"].iloc[0]
            m[f"D{threshold}cc"] = cc_val

        metrics.append(m)

    return pd.DataFrame(metrics)
