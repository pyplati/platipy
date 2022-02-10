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

from platipy.imaging.utils.crop import label_to_roi, crop_to_roi


def compute_volume(label):
    """Computes the volume in cubic centimetres

    Args:
        label (SimpleITK.Image): A binary mask.

    Returns:
        float: The volume (in cubic centimetres)
    """

    return sitk.GetArrayFromImage(label).sum() * np.product(label.GetSpacing()) / 1000


def compute_surface_metrics(label_a, label_b, verbose=False):
    """Compute surface distance metrics between two labels. Surface metrics computed are:
    hausdorffDistance, meanSurfaceDistance, medianSurfaceDistance, maximumSurfaceDistance,
    sigmaSurfaceDistance

    Args:
        label_a (sitk.Image): A mask to compare
        label_b (sitk.Image): Another mask to compare
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        dict: Dictionary object containing surface distance metrics
    """

    hausdorff_distance = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance.Execute(label_a, label_b)
    hd = hausdorff_distance.GetHausdorffDistance()

    mean_sd_list = []
    max_sd_list = []
    std_sd_list = []
    median_sd_list = []
    num_points = []
    for (la, lb) in ((label_a, label_b), (label_b, label_a)):

        label_intensity_stat = sitk.LabelIntensityStatisticsImageFilter()
        reference_distance_map = sitk.Abs(
            sitk.SignedMaurerDistanceMap(la, squaredDistance=False, useImageSpacing=True)
        )
        moving_label_contour = sitk.LabelContour(lb)
        label_intensity_stat.Execute(moving_label_contour, reference_distance_map)

        mean_sd_list.append(label_intensity_stat.GetMean(1))
        max_sd_list.append(label_intensity_stat.GetMaximum(1))
        std_sd_list.append(label_intensity_stat.GetStandardDeviation(1))
        median_sd_list.append(label_intensity_stat.GetMedian(1))

        num_points.append(label_intensity_stat.GetNumberOfPixels(1))

    if verbose:
        print("        Boundary points:  {0}  {1}".format(num_points[0], num_points[1]))

    mean_surf_dist = np.dot(mean_sd_list, num_points) / np.sum(num_points)
    max_surf_dist = np.max(max_sd_list)
    std_surf_dist = np.sqrt(
        np.dot(
            num_points,
            np.add(np.square(std_sd_list), np.square(np.subtract(mean_sd_list, mean_surf_dist))),
        )
    )
    median_surf_dist = np.mean(median_sd_list)

    result = {}
    result["hausdorffDistance"] = hd
    result["meanSurfaceDistance"] = mean_surf_dist
    result["medianSurfaceDistance"] = median_surf_dist
    result["maximumSurfaceDistance"] = max_surf_dist
    result["sigmaSurfaceDistance"] = std_surf_dist

    return result


def compute_volume_metrics(label_a, label_b):
    """Compute volume metrics between two labels. Volume metrics computed are:
    DSC, volumeOverlap fractionOverlap truePositiveFraction trueNegativeFraction
    falsePositiveFraction falseNegativeFraction

    Args:
        label_a (sitk.Image): A mask to compare
        label_b (sitk.Image): Another mask to compare

    Returns:
        dict: Dictionary object containing volume metrics
    """

    arr_a = sitk.GetArrayFromImage(label_a).astype(bool)
    arr_b = sitk.GetArrayFromImage(label_b).astype(bool)

    arr_intersection = arr_a & arr_b
    arr_union = arr_a | arr_b

    voxel_volume = np.product(label_a.GetSpacing()) / 1000.0  # Conversion to cm^3

    # 2|A & B|/(|A|+|B|)
    dsc = (2.0 * arr_intersection.sum()) / (arr_a.sum() + arr_b.sum())

    #  |A & B|/|A | B|
    frac_overlap = arr_intersection.sum() / arr_union.sum().astype(float)
    vol_overlap = arr_intersection.sum() * voxel_volume

    true_pos = arr_intersection.sum()
    true_neg = (np.invert(arr_a) & np.invert(arr_b)).sum()
    false_pos = arr_b.sum() - true_pos
    false_neg = arr_a.sum() - true_pos

    true_pos_frac = (1.0 * true_pos) / (true_pos + false_neg)
    true_neg_frac = (1.0 * true_neg) / (true_neg + false_pos)
    false_pos_frac = (1.0 * false_pos) / (true_neg + false_pos)
    false_neg_frac = (1.0 * false_neg) / (true_pos + false_neg)

    result = {}
    result["DSC"] = dsc
    result["volumeOverlap"] = vol_overlap
    result["fractionOverlap"] = frac_overlap
    result["truePositiveFraction"] = true_pos_frac
    result["trueNegativeFraction"] = true_neg_frac
    result["falsePositiveFraction"] = false_pos_frac
    result["falseNegativeFraction"] = false_neg_frac

    return result


def compute_metric_dsc(label_a, label_b, auto_crop=True):
    """Compute the Dice Similarity Coefficient between two labels

    Args:
        label_a (sitk.Image): A mask to compare
        label_b (sitk.Image): Another mask to compare

    Returns:
        float: The Dice Similarity Coefficient
    """
    if auto_crop:
        largest_region = (label_a + label_b) > 0
        crop_box_size, crop_box_index = label_to_roi(largest_region)

        label_a = crop_to_roi(label_a, size=crop_box_size, index=crop_box_index)
        label_b = crop_to_roi(label_b, size=crop_box_size, index=crop_box_index)

    arr_a = sitk.GetArrayFromImage(label_a).astype(bool)
    arr_b = sitk.GetArrayFromImage(label_b).astype(bool)
    return 2 * ((arr_a & arr_b).sum()) / (arr_a.sum() + arr_b.sum())


def compute_metric_specificity(label_a, label_b, auto_crop=True):
    """Compute the specificity between two labels

    Args:
        label_a (sitk.Image): A mask to compare
        label_b (sitk.Image): Another mask to compare

    Returns:
        float: The specificity between the two labels
    """
    if auto_crop:
        largest_region = (label_a + label_b) > 0
        crop_box_size, crop_box_index = label_to_roi(largest_region)

        label_a = crop_to_roi(label_a, size=crop_box_size, index=crop_box_index)
        label_b = crop_to_roi(label_b, size=crop_box_size, index=crop_box_index)

    arr_a = sitk.GetArrayFromImage(label_a).astype(bool)
    arr_b = sitk.GetArrayFromImage(label_b).astype(bool)

    arr_intersection = arr_a & arr_b

    true_pos = arr_intersection.sum()
    true_neg = (np.invert(arr_a) & np.invert(arr_b)).sum()
    false_pos = arr_b.sum() - true_pos

    return float((1.0 * true_neg) / (true_neg + false_pos))


def compute_metric_sensitivity(label_a, label_b, auto_crop=True):
    """Compute the sensitivity between two labels

    Args:
        label_a (sitk.Image): A mask to compare
        label_b (sitk.Image): Another mask to compare

    Returns:
        float: The sensitivity between the two labels
    """
    if auto_crop:
        largest_region = (label_a + label_b) > 0
        crop_box_size, crop_box_index = label_to_roi(largest_region)

        label_a = crop_to_roi(label_a, size=crop_box_size, index=crop_box_index)
        label_b = crop_to_roi(label_b, size=crop_box_size, index=crop_box_index)

    arr_a = sitk.GetArrayFromImage(label_a).astype(bool)
    arr_b = sitk.GetArrayFromImage(label_b).astype(bool)

    arr_intersection = arr_a & arr_b

    true_pos = arr_intersection.sum()
    false_neg = arr_a.sum() - true_pos

    return float((1.0 * true_pos) / (true_pos + false_neg))


def compute_metric_masd(label_a, label_b, auto_crop=True):
    """Compute the mean absolute distance between two labels

    Args:
        label_a (sitk.Image): A mask to compare
        label_b (sitk.Image): Another mask to compare

    Returns:
        float: The mean absolute surface distance
    """
    if auto_crop:
        largest_region = (label_a + label_b) > 0
        crop_box_size, crop_box_index = label_to_roi(largest_region)

        label_a = crop_to_roi(label_a, size=crop_box_size, index=crop_box_index)
        label_b = crop_to_roi(label_b, size=crop_box_size, index=crop_box_index)

    if (
        sitk.GetArrayViewFromImage(label_a).sum() == 0
        or sitk.GetArrayViewFromImage(label_b).sum() == 0
    ):
        return np.nan

    mean_sd_list = []
    num_points = []
    for (la, lb) in ((label_a, label_b), (label_b, label_a)):

        label_intensity_stat = sitk.LabelIntensityStatisticsImageFilter()
        reference_distance_map = sitk.Abs(
            sitk.SignedMaurerDistanceMap(la, squaredDistance=False, useImageSpacing=True)
        )
        moving_label_contour = sitk.LabelContour(lb)
        label_intensity_stat.Execute(moving_label_contour, reference_distance_map)

        mean_sd_list.append(label_intensity_stat.GetMean(1))
        num_points.append(label_intensity_stat.GetNumberOfPixels(1))

    mean_surf_dist = np.dot(mean_sd_list, num_points) / np.sum(num_points)
    return float(mean_surf_dist)


def compute_metric_hd(label_a, label_b, auto_crop=True):
    """Compute the Hausdorff distance between two labels

    Args:
        label_a (sitk.Image): A mask to compare
        label_b (sitk.Image): Another mask to compare

    Returns:
        float: The maximum Hausdorff distance
    """

    if auto_crop:
        largest_region = (label_a + label_b) > 0
        crop_box_size, crop_box_index = label_to_roi(largest_region)

        label_a = crop_to_roi(label_a, size=crop_box_size, index=crop_box_index)
        label_b = crop_to_roi(label_b, size=crop_box_size, index=crop_box_index)

    if (
        sitk.GetArrayViewFromImage(label_a).sum() == 0
        or sitk.GetArrayViewFromImage(label_b).sum() == 0
    ):
        return np.nan

    hausdorff_distance = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance.Execute(label_a, label_b)
    hausdorff_distance_value = hausdorff_distance.GetHausdorffDistance()

    return hausdorff_distance_value
