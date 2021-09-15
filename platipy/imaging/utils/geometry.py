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


def vector_angle(v1, v2, smallest=True):
    """Return the angle between two vectors

    Args:
        v1 (np.array): A three-dimensional vector
        v2 (np.array): A three-dimensional vector
        smallest (bool, optional): If True, the angle is the smallest (i.e. ignoring direction).
            If False, direction is taken into account, so angle can be obtuse. Defaults to True.

    Returns:
        float: The angle in radians
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    dot_product = np.dot(v1_norm, v2_norm)
    if smallest:
        dot_product = np.abs(dot_product)
    angle = np.arccos(dot_product)
    return angle


def rotate_image(
    img,
    rotation_centre=(0, 0, 0),
    rotation_axis=(1, 0, 0),
    rotation_angle_radians=0,
    interpolation=sitk.sitkNearestNeighbor,
    default_value=0,
):
    """Rotates an image

    Args:
        img (SimpleITK.Image): The image to rotate
        rotation_centre (tuple, optional): The centre of rotation (in physical coordinates).
            Defaults to (0, 0, 0).
        rotation_axis (tuple, optional): The axis of rotation. Defaults to (1, 0, 0).
        rotation_angle_radians (float, optional): The angle of rotation. Defaults to 0.
        interpolation (int, optional): Final interpolation. Defaults to sitk.sitkNearestNeighbor.
        default_value (int, optional): Default value. Defaults to 0.

    Returns:
        SimpleITK.Image: The rotated image, resampled into the original space.
    """

    # Define the transform, using predefined centre of rotation and given angle
    rotation_transform = sitk.VersorRigid3DTransform()
    rotation_transform.SetCenter(rotation_centre)
    rotation_transform.SetRotation(rotation_axis, rotation_angle_radians)

    # Resample the image using the rotation transform
    resampled_image = sitk.Resample(
        img,
        rotation_transform,
        interpolation,
        default_value,
        img.GetPixelID(),
    )

    return resampled_image
