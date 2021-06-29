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
    """Insert a sphere into the given array

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

    if not hasattr(sp_radius, "__iter__"):
        sp_radius = [sp_radius] * 3

    sp_radius_x, sp_radius_y, sp_radius_z = sp_radius

    arr_copy[
        ((x - sp_centre[0]) / sp_radius_x) ** 2.0
        + ((y - sp_centre[1]) / sp_radius_y) ** 2.0
        + ((z - sp_centre[2]) / sp_radius_z) ** 2.0
        <= 1
    ] = 1

    return arr_copy


def insert_cylinder(arr, cyl_radius=4, cyl_height=2, cyl_centre=(0, 0, 0)):
    """
    Insert a cylinder into the given array.
    The cylinder vertical extent is +/- 0.5 * height

    Args:
        arr (np.ndarray): The array into which the cylinder is inserted
        cyl_radius (int, optional): Cylinder radius. Defaults to 4.
        cyl_height (int, optional): Cylinder height. Defaults to 2.
        cyl_centre (tuple, optional): Cylinder centre. Defaults to (0, 0, 0).

    Returns:
        np.ndarray: The original array with a cylinder (value 1)
    """
    arr_copy = arr[:]

    x, y, z = np.indices(arr.shape)

    if not hasattr(cyl_radius, "__iter__"):
        cyl_radius = [cyl_radius] * 2

    condition_radial = (
        ((z - cyl_centre[0]) / cyl_radius[0]) ** 2 + ((y - cyl_centre[1]) / cyl_radius[1]) ** 2
    ) <= 1
    condition_height = np.abs((x - cyl_centre[2]) / (0.5 * cyl_height)) <= 1

    arr_copy[condition_radial & condition_height] = 1

    return arr_copy


def insert_sphere_image(image, sp_radius, sp_centre):
    """Insert a sphere into an image

    Args:
        image (sitk.Image): Image in which to insert sphere
        sp_radius (int | list, optional): The radius of the sphere.
            Can also be defined as a vector. Defaults to 4.
        sp_centre (tuple, optional): The position at which the sphere should be inserted. Defaults
            to (0, 0, 0).

    Returns:
        np.array: An array with the sphere inserted
    """

    if not hasattr(sp_radius, "__iter__"):
        sp_radius = [sp_radius] * 3

    sp_radius_image = [i / j for i, j in zip(sp_radius, image.GetSpacing()[::-1])]

    arr = sitk.GetArrayFromImage(image)

    arr = insert_sphere(arr, sp_radius_image, sp_centre)

    image_sphere = sitk.GetImageFromArray(arr)
    image_sphere.CopyInformation(image)

    return image_sphere


def insert_cylinder_image(image, cyl_radius=(5, 5), cyl_height=10, cyl_centre=(0, 0, 0)):
    """Insert a cylinder into an image

    Args:
        image (SimpleITK.Image):
        cyl_radius (tuple, optional): Cylinder radius, can be defined as a single value
            or a tuple (will generate an ellipsoid). Defaults to (5,5).
        cyl_height (int, optional): Cylinder height. Defaults to 10.
        cyl_centre (tuple, optional): Cylinder centre. Defaults to (0,0,0).

    Returns:
        SimpleITK.Image: Image with cylinder inserted
    """
    if not hasattr(cyl_radius, "__iter__"):
        cyl_radius = [cyl_radius] * 2

    cyl_radius_image = [i / j for i, j in zip(cyl_radius, image.GetSpacing()[1::-1])]
    cyl_height_image = cyl_height / image.GetSpacing()[2]

    arr = sitk.GetArrayFromImage(image)

    arr = insert_cylinder(arr, cyl_radius_image, cyl_height_image, cyl_centre)

    image_cylinder = sitk.GetImageFromArray(arr)
    image_cylinder.CopyInformation(image)

    return image_cylinder
