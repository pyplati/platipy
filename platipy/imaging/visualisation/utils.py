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

from matplotlib import rcParams
from skimage.color import hsv2rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable  # , AxesGrid, ImageGrid

import warnings


import math
import pathlib
import numpy as np
import SimpleITK as sitk

def return_slice(axis, index):
    """Prepares a slice tuple to use for extracting a slice for rendering

    Args:
        axis (str): One of "x", "y" or "z"
        index (int): The index of the slice to fetch

    Returns:
        tuple: can be used to extract a slice
    """

    if axis == "x":
        return (slice(None), slice(None), index)
    if axis == "y":
        return (slice(None), index, slice(None))
    if axis == "z":
        return (index, slice(None), slice(None))

    return None

def subsample_vector_field(axis, cut, subsample=1):
"""Prepares a slice tuple to use for extracting a slice for rendering

Args:
    axis (str): One of "x", "y" or "z"
    cut (int): The index of the image slice
    subsample (int): the subsample factor

Returns:
    tuple: can be used to extract a vector field slice
"""
if hasattr(subsample, "__iter__"):
    subsample_ax, subsample_cor, subsample_sag = subsample
else:
    subsample_ax, subsample_cor, subsample_sag = (subsample,) * 3

if axis == "x":
    return (
        slice(None, None, subsample_ax),
        slice(None, None, subsample_cor),
        cut,
    )
if axis == "y":
    return (
        slice(None, None, subsample_ax),
        cut,
        slice(None, None, subsample_sag),
    )
if axis == "z":
    return (
        cut,
        slice(None, None, subsample_cor),
        slice(None, None, subsample_sag),
    )
return None


def vector_image_grid(axis, vector_field_array, subsample=1):
    """Prepares a grid for rendering a vector field on an image

    Args:
        axis (str): One of "x", "y" or "z"
        vector_field_array (np.array): the vector field array
        subsample (int): the subsample factor

    Returns:
        tuple: defines the 2 dimensional grid for displaying vectors
    """
    if hasattr(subsample, "__iter__"):
        subsample_ax, subsample_cor, subsample_sag = subsample
    else:
        subsample_ax, subsample_cor, subsample_sag = (subsample,) * 3

    if axis == "x":
        return np.mgrid[
            0 : vector_field_array.shape[1] : subsample_cor,
            0 : vector_field_array.shape[0] : subsample_ax,
        ]
    if axis == "y":
        return np.mgrid[
            0 : vector_field_array.shape[2] : subsample_sag,
            0 : vector_field_array.shape[0] : subsample_ax,
        ]
    if axis == "z":
        return np.mgrid[
            0 : vector_field_array.shape[2] : subsample_sag,
            0 : vector_field_array.shape[1] : subsample_cor,
        ]
    return None

def reorientate_vector_field(axis, vector_ax, vector_cor, vector_sag, invert_field=True):
    """Reorients vector field components for rendering
    This is necessary after converting from sitk.Image to np.array

    Args:
        axis (str): One of "x", "y" or "z"
        vector_ax (np.array): The first vector component (z)
        vector_cor (np.array): The second vector component (y)
        vector_sag (np.array): The third vector component (x)

    Returns:
        tuple: the re-oriented vector field components
    """

    if invert_field:
        vector_ax = -vector_ax
        vector_cor = -vector_cor
        vector_sag = -vector_sag

    if axis == "x":  # sagittal projection
        return vector_cor, vector_ax, vector_sag
    if axis == "y":  # coronal projection
        return vector_sag, vector_ax, vector_cor
    if axis == "z":  # axial projection
        return vector_sag, -vector_cor, vector_ax

    return None

def generate_comparison_colormix(image_list, arr_slice=None, window=(-250, 500), color_rotation=0.35):


    if all( (isinstance(image, sitk.Image), for image in image_list) ):
        if any (image.GetDimension() >= 3), for image in image_list) ) and arr_slice==None:
            raise ValueError("Images cannot be 3D unless 'arr_slice' is specified.")
            
        else:
            array_list = [sitk.GetArrayViewFromImage(image).__getitem__(arr_slice) for image in image_list]

    elif all ( (isinstance(image, np.ndarray), for image in image_list) ):
        if any (len(image.shape) >= 3), for image in image_list) ) and arr_slice==None:
            raise ValueError("Images cannot be 3D unless 'arr_slice' is specified.")

        else:
            array_list = [image.__getitem__(arr_slice) for image in image_list]

            nda_a = nda_original.__getitem__(s_ax)
            nda_b = nda_new.__getitem__(s_ax)

            nda_a_norm = (np.clip(nda_a, window[0], window[0] + window[1]) - window[0]) / (
                window[1]
            )
            nda_b_norm = (np.clip(nda_b, window[0], window[0] + window[1]) - window[0]) / (
                window[1]
            )

            nda_color = np.stack(
                [
                    color_rotation * (nda_a_norm > nda_b_norm)
                    + (0.5 + color_rotation) * (nda_a_norm <= nda_b_norm),
                    np.abs(nda_a_norm - nda_b_norm),
                    (nda_a_norm + nda_b_norm) / 2,
                ],
                axis=-1,
            )

            ax_ax.imshow(
                hsv2rgb(nda_color),