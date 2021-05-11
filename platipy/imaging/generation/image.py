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


def insert_sphere(arr, sp_radius=4, sp_centre=(0, 0, 0)):
    """Insert a sphere into the give array

    Args:
        arr (np.array): Array in which to insert sphere
        sp_radius (int, optional): The radius of the sphere. Defaults to 4.
        sp_centre (tuple, optional): The position at which the sphere should be inserted. Defaults
                                     to (0, 0, 0).

    Returns:
        np.array: An array with the sphere inserted
    """

    arr_copy = arr[:]

    x, y, z = np.indices(arr.shape)

    if hasattr(sp_radius, "__iter__"):
        sp_radius_x, sp_radius_y, sp_radius_z = sp_radius

    arr_copy[
        ((x - sp_centre[0]) / sp_radius_x) ** 2
        + ((y - sp_centre[1]) / sp_radius_y) ** 2
        + ((z - sp_centre[2]) / sp_radius_z) ** 2
        <= 1
    ] = 1

    return arr_copy


def insert_sphere_image(image, sp_radius, sp_centre):
    """Insert a sphere into a blank image with the same size as image

    Args:
        image (sitk.Image): Image in which to insert sphere
        sp_radius (int, optional): The radius of the sphere. Defaults to 4.
        sp_centre (tuple, optional): The position at which the sphere should be inserted. Defaults
                                     to (0, 0, 0).

    Returns:
        np.array: An array with the sphere inserted
    """

    sp_radius_image = [sp_radius * i for i in image.GetSpacing()]

    arr = sitk.GetArrayFromImage(image)

    arr = insert_sphere(arr, sp_radius_image, sp_centre)

    image_sphere = sitk.GetImageFromArray(arr)
    image_sphere.CopyInformation(image)

    return image_sphere
