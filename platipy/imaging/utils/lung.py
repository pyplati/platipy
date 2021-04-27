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
    bmcif.SetKernelType(sitk.sitkBall)
    bmcif.SetKernelRadius(kernel_radius)

    # Save the largest as the external contour
    external_mask = sitk.BinaryThreshold(label_image, labels[0]["label"], labels[0]["label"])
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
    bmcif.SetKernelType(sitk.sitkBall)
    bmcif.SetKernelRadius(kernel_radius)

    # Save the 2nd largest as the lung mask
    lung_mask = sitk.BinaryThreshold(
        label_image, labels[lung_idx]["label"], labels[lung_idx]["label"]
    )
    lung_mask = bmcif.Execute(lung_mask)

    return lung_mask


def fill_holes(img, label_image, external_mask, lung_mask):
    # Fill all other holes
    img_array = sitk.GetArrayFromImage(img)

    bdif = sitk.BinaryDilateImageFilter()
    bdif.SetKernelType(sitk.sitkBall)
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