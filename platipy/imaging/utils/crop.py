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

        bounding_box[3 + i] = min(bounding_box[3 + i], img.GetSize()[i] - bounding_box[i])

    return bounding_box


def label_to_roi(label, expansion_mm=[0, 0, 0]):
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

    return crop_box_size, crop_box_index


def crop_to_roi(image, size, index):
    """Utility function for cropping images"""
    return sitk.RegionOfInterest(image, size=size, index=index)
