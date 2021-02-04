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


"""
Various useful utility functions for atlas based segmentation algorithms.
"""
import itk

import numpy as np
import SimpleITK as sitk

from scipy.stats import norm as scipy_norm
from scipy.ndimage.measurements import center_of_mass


def get_com(label, as_int=True):
    arr = sitk.GetArrayFromImage(label)
    com = center_of_mass(arr)

    if as_int:
        com = [int(i) for i in com]

    return com


def vectorised_transform_index_to_physical_point(image, point_array, rotate=True):
    """
    Transforms a set of points from array indices to real-space
    """
    if rotate:
        spacing = image.GetSpacing()[::-1]
        origin = image.GetOrigin()[::-1]
    else:
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
    return point_array * spacing + origin


def vectorised_transform_physical_point_to_index(image, point_array, rotate=True):
    """
    Transforms a set of points from real-space to array indices
    """
    if rotate:
        spacing = image.GetSpacing()[::-1]
        origin = image.GetOrigin()[::-1]
    else:
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
    return (point_array - origin) / spacing


def median_absolute_deviation(data, axis=None):
    """Median Absolute Deviation: a "Robust" version of standard deviation.
    Indices variabililty of the sample.
    https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    return np.median(np.abs(data - np.median(data, axis=axis)), axis=axis)


def gaussian_curve(x, a, m, s):
    return a * scipy_norm.pdf(x, loc=m, scale=s)


def sitk_to_itk(sitk_image, copy_info=True):
    """
    Helper function to convert SimpleITK images to ITK images
    """
    sitk_arr = sitk.GetArrayFromImage(sitk_image)

    itk_image = itk.GetImageFromArray(sitk_arr, is_vector=False)
    if copy_info:
        itk_image.SetOrigin(sitk_image.GetOrigin())
        itk_image.SetSpacing(sitk_image.GetSpacing())
        itk_image.SetDirection(
            itk.GetMatrixFromArray(
                np.reshape(np.array(sitk_image.GetDirection()), [3] * 2)
            )
        )

    return itk_image


def itk_to_sitk(itk_image):
    """
    Helper function to convert ITK images to SimpleITK images
    """
    sitk_image = sitk.GetImageFromArray(
        itk.GetArrayFromImage(itk_image), isVector=False
    )
    sitk_image.SetOrigin(tuple(itk_image.GetOrigin()))
    sitk_image.SetSpacing(tuple(itk_image.GetSpacing()))
    sitk_image.SetDirection(itk.GetArrayFromMatrix(itk_image.GetDirection()).flatten())

    return sitk_image


def detect_holes(img, lower_threshold=-10000, upper_threshold=-400):
    """
    Detect all (air) holes in given image. Default threshold values given are defined for CT.

    Returns:
        label_image: image containing labels of all holes detected
        labels: a list of holes detected and some of their properties
    """

    # Threshold to get holes
    btif = sitk.BinaryThresholdImageFilter()
    btif.SetInsideValue(1)
    btif.SetOutsideValue(0)
    btif.SetLowerThreshold(lower_threshold)
    btif.SetUpperThreshold(upper_threshold)
    holes = btif.Execute(img)

    # Get connected components
    ccif = sitk.ConnectedComponentImageFilter()
    label_image = ccif.Execute(holes, False)

    labels = []

    lssif = sitk.LabelShapeStatisticsImageFilter()
    lssif.Execute(label_image)
    for region in range(1, ccif.GetObjectCount()):
        label = {
            "label": region,
            "phys_size": lssif.GetPhysicalSize(region),
            "elongation": lssif.GetElongation(region),
            "roundness": lssif.GetRoundness(region),
            "perimeter": lssif.GetPerimeter(region),
            "flatness": lssif.GetFlatness(region),
        }
        labels.append(label)

    # Sort by size
    labels = sorted(labels, key=lambda i: i["phys_size"], reverse=True)

    return label_image, labels


def get_external_mask(label_image, labels, kernel_radius=5):
    """
    Gets the external mask based on the label image and labels generated using detect_holes
    """

    bmcif = sitk.BinaryMorphologicalClosingImageFilter()
    bmcif.SetKernelType(bmcif.Ball)
    bmcif.SetKernelRadius(kernel_radius)

    # Save the largest as the external contour
    external_mask = sitk.BinaryThreshold(
        label_image, labels[0]["label"], labels[0]["label"]
    )
    external_mask = bmcif.Execute(external_mask)

    return external_mask


def get_lung_mask(label_image, labels, kernel_radius=2):

    lung_idx = 1
    while labels[lung_idx]["flatness"] > 2:
        lung_idx += 1

        if lung_idx > len(labels):
            print("Flatness not satisfied!")
            return None

    bmcif = sitk.BinaryMorphologicalClosingImageFilter()
    bmcif.SetKernelType(bmcif.Ball)
    bmcif.SetKernelRadius(kernel_radius)

    # Save the 2nd largest as the lung mask
    lung_mask = sitk.BinaryThreshold(
        label_image, labels[lung_idx]["label"], labels[lung_idx]["label"]
    )
    lung_mask = bmcif.Execute(lung_mask)

    return lung_mask


def fill_holes(img, label_image, labels, external_mask, lung_mask):
    # Fill all other holes
    img_array = sitk.GetArrayFromImage(img)

    bdif = sitk.BinaryDilateImageFilter()
    bdif.SetKernelType(bdif.Ball)
    bdif.SetKernelRadius(3)

    mask = sitk.BinaryThreshold(label_image, 1, int(img_array.max()))
    mask = sitk.Subtract(mask, external_mask)
    mask = sitk.Subtract(mask, lung_mask)
    mask = bdif.Execute(mask)

    mask_array = sitk.GetArrayFromImage(mask)
    img_array[mask_array == 1] = 50

    filled_img = sitk.GetImageFromArray(img_array)
    filled_img.CopyInformation(img)

    return filled_img


def get_crop_bounding_box(img, mask):

    label_shape_analysis = sitk.LabelShapeStatisticsImageFilter()
    label_shape_analysis.Execute(mask)
    mask_box = label_shape_analysis.GetBoundingBox(True)

    sag_padding = 50
    cor_padding = 50
    ax_padding = 30
    ax_extent = 250

    phys_bb_origin = list(mask.TransformIndexToPhysicalPoint(mask_box[0:3]))
    phys_bb_origin[0] -= sag_padding
    phys_bb_origin[1] -= cor_padding
    phys_bb_origin[2] -= ax_extent - ax_padding
    bb_origin = img.TransformPhysicalPointToIndex(phys_bb_origin)

    phys_bb_size = [0, 0, 0]
    bb_size = [0, 0, 0]
    for i in range(3):
        phys_bb_size[i] = mask_box[3 + i] * mask.GetSpacing()[i]
        if i == 0:
            phys_bb_size[i] += sag_padding * 2
        if i == 1:
            phys_bb_size[i] += cor_padding * 2
        if i == 2:
            phys_bb_size[i] = ax_extent + ax_padding * 2
        bb_size[i] = phys_bb_size[i] / mask.GetSpacing()[i]

    bounding_box = bb_origin + tuple(bb_size)
    bounding_box = [int(i) for i in bounding_box]

    for i in range(3):
        if bounding_box[i] < 0:
            bounding_box[3 + i] = bounding_box[3 + i] + bounding_box[i]
            bounding_box[i] = max(bounding_box[i], 0)

        bounding_box[3 + i] = min(
            bounding_box[3 + i], img.GetSize()[i] - bounding_box[i]
        )

    return bounding_box


def label_to_roi(image, label_list, expansion=[0, 0, 0]):

    label_stats_image_filter = sitk.LabelStatisticsImageFilter()
    if type(label_list) == list:
        label_stats_image_filter.Execute(image, sum(label_list) > 0)
    elif type(label_list) == sitk.Image:
        label_stats_image_filter.Execute(image, label_list)
    else:
        raise ValueError("Second argument must be a SITK image, or list thereof.")

    bounding_box = np.array(label_stats_image_filter.GetBoundingBox(1))

    index = [bounding_box[x * 2] for x in range(3)]
    size = [bounding_box[(x * 2) + 1] - bounding_box[x * 2] for x in range(3)]
    expansion = np.array(expansion)

    # Avoid starting outside the image
    crop_box_index = np.max([index - expansion, np.array([0, 0, 0])], axis=0)

    # Avoid ending outside the image
    crop_box_size = np.min(
        [
            np.array(image.GetSize()) - crop_box_index,
            np.array(size) + 2 * expansion,
        ],
        axis=0,
    )

    crop_box_size = [int(i) for i in crop_box_size]
    crop_box_index = [int(i) for i in crop_box_index]

    return crop_box_size, crop_box_index


def crop_to_roi(image, size, index):
    return sitk.RegionOfInterest(image, size=size, index=index)
