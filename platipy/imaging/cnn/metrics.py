import collections
import math

import SimpleITK as sitk
import numpy as np

from platipy.imaging.label.comparison import compute_surface_dsc, compute_metric_dsc
from platipy.imaging.label.utils import get_union_mask, get_intersection_mask


def probabilistic_dice(gt_labels, sampled_labels, dsc_type="dsc", tau=3):

    gt_union = get_union_mask(gt_labels)
    gt_intersection = get_intersection_mask(gt_labels)

    st_union = get_union_mask(sampled_labels)
    st_intersection = get_intersection_mask(sampled_labels)

    if dsc_type == "dsc":
        dsc_union = compute_metric_dsc(gt_union, st_union)
        dsc_intersection = compute_metric_dsc(gt_intersection, st_intersection)

    if dsc_type == "sdsc":
        dsc_union = compute_surface_dsc(gt_union, st_union, tau=tau)
        dsc_intersection = compute_surface_dsc(gt_intersection, st_intersection, tau=tau)

    return (dsc_union + dsc_intersection) / 2


def probabilistic_surface_dice(gt_labels, sampled_labels, sd_range=3, tau=0):

    if isinstance(gt_labels, dict):
        gt_labels = [gt_labels[l] for l in gt_labels]

    if isinstance(sampled_labels, dict):
        sampled_labels = [sampled_labels[l] for l in sampled_labels]

    binary_contour_filter = sitk.BinaryContourImageFilter()
    binary_contour_filter.FullyConnectedOff()
    summed = None
    for mask in gt_labels:

        if summed is None:
            summed = mask

        else:
            summed += mask

    intersection = summed >= 1
    union = summed >= 5

    mask_mean = summed >= 3
    intersection_minus_mean = intersection - mask_mean
    mean_minus_union = mask_mean - union

    contour_i = binary_contour_filter.Execute(intersection)
    contour_u = binary_contour_filter.Execute(union)
    contour_mean = binary_contour_filter.Execute(mask_mean)

    dist_to_i = sitk.SignedMaurerDistanceMap(
        contour_i, useImageSpacing=True, squaredDistance=False
    )

    dist_to_u = sitk.SignedMaurerDistanceMap(
        contour_u, useImageSpacing=True, squaredDistance=False
    )

    dist_to_mean = sitk.SignedMaurerDistanceMap(
        contour_mean, useImageSpacing=True, squaredDistance=False
    )

    mean = 0
    sd = 1 / sd_range
    max_agg = np.pi * sd

    dist_sum = dist_to_mean + dist_to_i
    dist_ratio_neg = dist_to_mean / dist_sum

    dist_ratio_arr = sitk.GetArrayFromImage(dist_ratio_neg)

    dist_ratio_arr = (np.pi * sd) * np.exp(-0.5 * ((dist_ratio_arr - mean) / sd) ** 2)
    dist_ratio_arr = dist_ratio_arr / max_agg / 2  # Normalise
    dist_ratio_arr[sitk.GetArrayFromImage(intersection_minus_mean) == 0] = 0
    dist_ratio_neg = sitk.GetImageFromArray(dist_ratio_arr)
    dist_ratio_neg.CopyInformation(dist_sum)

    dist_sum = dist_to_mean + dist_to_u
    dist_ratio_pos = dist_to_u / dist_sum

    dist_ratio_arr = sitk.GetArrayFromImage(dist_ratio_pos)

    dist_ratio_arr = (np.pi * sd) * np.exp(-0.5 * ((dist_ratio_arr - mean) / sd) ** 2)
    dist_ratio_arr = (dist_ratio_arr / max_agg / 2) + 0.5  # Normalise
    dist_ratio_arr[sitk.GetArrayFromImage(mean_minus_union) == 0] = 0
    dist_ratio_arr[sitk.GetArrayFromImage(union) == 1] = 1
    dist_ratio_pos = sitk.GetImageFromArray(dist_ratio_arr)
    dist_ratio_pos.CopyInformation(dist_sum)

    dist_ratio = dist_ratio_neg + dist_ratio_pos

    sample_count = math.floor(len(sampled_labels) / 2)

    ranges = {}
    range_masks = {}
    start_mask = None
    for pr in np.linspace(0.5, 1, sample_count + 1):
        next_mask = dist_ratio >= pr
        next_contour = binary_contour_filter.Execute(next_mask)

        if start_mask is None:
            ranges[pr] = next_contour
        else:
            ranges[pr] = ((start_mask - next_mask) + start_contour + next_contour) > 0

        range_masks[pr] = next_mask

        start_mask = next_mask
        start_contour = binary_contour_filter.Execute(start_mask)

    start_mask = None
    for pr in np.linspace(0.5, 0.000001, sample_count + 1):
        next_mask = dist_ratio >= pr
        next_contour = binary_contour_filter.Execute(next_mask)

        if start_mask is None:
            ranges[pr] = next_contour
        else:
            ranges[pr] = ((next_mask - start_mask) + start_contour + next_contour) > 0

        range_masks[pr] = next_mask

        start_mask = next_mask
        start_contour = binary_contour_filter.Execute(start_mask)

    ranges = collections.OrderedDict(sorted(ranges.items()))
    range_masks = collections.OrderedDict(sorted(range_masks.items()))

    result = 0
    for idx, r in enumerate(ranges):
        auto_mask = sampled_labels[idx]
        auto_contour = binary_contour_filter.Execute(auto_mask > 0)

        dist_to_range = sitk.SignedMaurerDistanceMap(
            ranges[r], useImageSpacing=True, squaredDistance=False
        )

        auto_intersection = sitk.GetArrayFromImage(auto_contour * (dist_to_range <= tau)).sum()

        this_result = auto_intersection / sitk.GetArrayFromImage(auto_contour).sum()
        if np.isnan(this_result):
            this_result = 0

        result += this_result

    return result / len(ranges)
