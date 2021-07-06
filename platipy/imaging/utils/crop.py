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

import numpy as np
import SimpleITK as sitk


def label_to_roi(label, expansion_mm=[0, 0, 0], return_as_list=False):
    """Generates a region of interest (ROI), defined by a starting index (z,y,x)
    and size (s_z, s_y, s_x). This can be used to crop images/labels.

    Args:
        label (sitk.Image | list): Binary label image/mask/structure to define to ROI
        expansion_mm (list, optional): An optional expansion of the box (in each direciton).
            Defaults to [0, 0, 0].

    Returns:
        tuple: crop_box_size, crop_box_starting_index
    """

    if hasattr(label, "__iter__") and not isinstance(label, sitk.Image):
        reference_label = sum(label) > 0
    else:
        reference_label = label > 0

    image_spacing = np.array(reference_label.GetSpacing())

    label_stats_image_filter = sitk.LabelStatisticsImageFilter()
    label_stats_image_filter.Execute(reference_label, reference_label)
    bounding_box = np.array(label_stats_image_filter.GetBoundingBox(1))

    index = [bounding_box[x * 2] for x in range(3)]
    size = [bounding_box[(x * 2) + 1] - bounding_box[x * 2] for x in range(3)]

    expansion_mm = np.array(expansion_mm)
    expansion = (expansion_mm / image_spacing).astype(int)

    # Avoid starting outside the image
    crop_box_index = np.max([index - expansion, np.array([0, 0, 0])], axis=0)

    # Avoid ending outside the image
    crop_box_size = np.min(
        [
            np.array(reference_label.GetSize()) - crop_box_index,
            np.array(size) + 2 * expansion,
        ],
        axis=0,
    )

    crop_box_size = [int(i) for i in crop_box_size]
    crop_box_index = [int(i) for i in crop_box_index]

    if return_as_list:
        return crop_box_index + crop_box_size

    return crop_box_size, crop_box_index


def crop_to_roi(image, size, index):
    """Utility function for cropping images"""
    return sitk.RegionOfInterest(image, size=size, index=index)


def crop_to_label_extent(image, label, expansion_mm=0):
    """Crop an image to the 3D extent defined by a binary mask (label).

    Args:
        image (SimpleITK.Image): The image to crop.
        label (SimpleITK.Image): The binary mask (label) defining the region to crop to.
        expansion_mm (float | tuple | list, optional): An optional expansion in physical units.
            Can be defined as a single number or iterable: (axial, coronal, sagittal) expansion.
            Defaults to 0.

    Returns:
        SimpleITK.Image: The cropped image.
    """

    if ~hasattr(expansion_mm, "__iter__"):
        expansion_mm = [
            expansion_mm,
        ] * 3

    cbox_s, cbox_i = label_to_roi(label, expansion_mm=expansion_mm)
    return crop_to_roi(image, cbox_s, cbox_i)
