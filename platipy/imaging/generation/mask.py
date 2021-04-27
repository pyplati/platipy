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

from skimage.morphology import convex_hull_image

import numpy as np
import SimpleITK as sitk


def get_bone_mask(image, lower_threshold=350, upper_threshold=3500, max_hole_size=5):
    """
    Automatically generate a binary mask of bones from a CT image.

    Args:
        image ([SimpleITK.Image]): The patient x-ray CT image to segment.
        lower_threshold (int, optional): Lower voxel value for threshold. Defaults to 350.
        upper_threshold (int, optional): Upper voxel value for threshold. Defaults to 3500.
        max_hole_size (int | list | bool, optional): Maximum hole size to be filled in millimetres.
                                                     Can be specified as a vector (z,y,x). Defaults
                                                     to 5.

    Returns:
        [SimpleITK.Image]: The binary bone mask.
    """

    bone_mask = sitk.BinaryThreshold(
        image, lowerThreshold=lower_threshold, upperThreshold=upper_threshold
    )

    if max_hole_size is not False:
        if not hasattr(max_hole_size, "__iter__"):
            max_hole_size = (max_hole_size,) * 3

    bone_mask = sitk.BinaryMorphologicalClosing(bone_mask, max_hole_size)

    return bone_mask


def get_external_mask(
    image, lower_threshold=-100, upper_threshold=2500, dilate=1, max_hole_size=False
):
    """
    Automatically generate a binary mask of the patient external contour.
    Uses slice-wise convex hull generation.

    Args:
        image ([SimpleITK.Image]): The patient x-ray CT image to segment. May work with other
                                   modalities with modified thresholds.
        lower_threshold (int, optional): Lower voxel value for threshold. Defaults to -100.
        upper_threshold (int, optional): Upper voxel value for threshold. Defaults to 2500.
        dilate (int | list | bool, optional): Dilation filter size applied to the binary mask. Can
                                              be specified as a vector (z,y,x). Defaults to 1.
        max_hole_size (int  | list | bool, optional): Maximum hole size to be filled in
                                                      millimetres. Can be specified as a vector
                                                      (z,y,x). Defaults to False.

    Returns:
        [SimpleITK.Image]: The binary external mask.
    """

    # Get all points inside the body
    external_mask = sitk.BinaryThreshold(
        image, lowerThreshold=lower_threshold, upperThreshold=upper_threshold
    )

    external_mask_components = sitk.ConnectedComponent(external_mask, True)

    # Second largest volume is most likely the body - you should check this!
    body_mask = sitk.Equal(sitk.RelabelComponent(external_mask_components), 1)

    if dilate is not False:
        if not hasattr(dilate, "__iter__"):
            dilate = (dilate,) * 3
        body_mask = sitk.BinaryDilate(body_mask, dilate)

    if max_hole_size is not False:
        if not hasattr(max_hole_size, "__iter__"):
            max_hole_size = (max_hole_size,) * 3

        body_mask = sitk.BinaryMorphologicalClosing(body_mask, max_hole_size)
        body_mask = sitk.BinaryFillhole(body_mask, fullyConnected=True)

    arr = sitk.GetArrayFromImage(body_mask)

    convex_hull_slices = np.zeros_like(arr)

    for index in np.arange(0, np.alen(arr)):
        convex_hull_slices[index] = convex_hull_image(arr[index])

    body_mask_hull = sitk.GetImageFromArray(convex_hull_slices)
    body_mask_hull.CopyInformation(body_mask)

    return body_mask_hull
