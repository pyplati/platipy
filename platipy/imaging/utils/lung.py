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

import SimpleITK as sitk


def detect_holes(img, lower_threshold=-10000, upper_threshold=-400):
    """
    Detect all (air) holes in given image. Default threshold values given are defined for CT.

    Args:
        img (sitk.Image): The image in which to detect holes.
        lower_threshold (int, optional): The lower threshold of intensities. Defaults to -10000.
        upper_threshold (int, optional): The upper threshold of intensities. Defaults to -400.

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
    label_image = ccif.Execute(holes)

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
    """Get a External mask given a label image and labels computed with detect_holes

    Args:
        label_image (sitk.Image): Label image from detect_holes
        labels (list): List of labels from detect_holes
        kernel_radius (int, optional): The width of the radius used by binary close. Defaults to 5.

    Returns:
        sitk.Image: External mask
    """

    bmcif = sitk.BinaryMorphologicalClosingImageFilter()
    bmcif.SetKernelType(sitk.sitkBall)
    bmcif.SetKernelRadius(kernel_radius)

    # Save the largest as the external contour
    external_mask = sitk.BinaryThreshold(label_image, labels[0]["label"], labels[0]["label"])
    external_mask = bmcif.Execute(external_mask)

    return external_mask


def get_lung_mask(label_image, labels, kernel_radius=2):
    """Get a Lung mask given a label image and labels computed with detect_holes

    Args:
        label_image (sitk.Image): Label image from detect_holes
        labels (list): List of labels from detect_holes
        kernel_radius (int, optional): The width of the radius used by binary close. Defaults to 2.

    Returns:
        sitk.Image: Lung mask
    """

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


def fill_holes(img, label_image, external_mask, lung_mask, fill_value=50):
    """Returns the input image with all holes filled (except for the external and lung holes).

    Args:
        img (sitk.Image): The image to fill
        label_image (sitk.Image): Label image from detect_holes
        external_mask (sitk.Image): The external mask of the patient.
        lung_mask (sitk.Image): The lung mask of the patient.
        fill_value (int): The value to use to fill the holes. Defaults to 50.

    Returns:
        sitk.Image: The image with holes filled
    """

    img_array = sitk.GetArrayFromImage(img)

    bdif = sitk.BinaryDilateImageFilter()
    bdif.SetKernelType(sitk.sitkBall)
    bdif.SetKernelRadius(3)

    mask = sitk.BinaryThreshold(label_image, 1, int(img_array.max()))
    mask = sitk.Subtract(mask, external_mask)
    mask = sitk.Subtract(mask, lung_mask)
    mask = bdif.Execute(mask)

    mask_array = sitk.GetArrayFromImage(mask)
    img_array[mask_array == 1] = fill_value

    filled_img = sitk.GetImageFromArray(img_array)
    filled_img.CopyInformation(img)

    return filled_img
