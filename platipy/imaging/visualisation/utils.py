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

from skimage.color import hsv2rgb

import numpy as np
import SimpleITK as sitk

import matplotlib.pyplot as plt

from platipy.imaging.utils.crop import label_to_roi


class VisualiseImage:
    """Class to represent visualisation of an image"""

    def __init__(self, image, aspect, interpolation, origin, colormap, clim):
        self.image = image
        self.aspect = aspect
        self.interpolation = interpolation
        self.origin = origin
        self.colormap = colormap
        self.clim = clim


class VisualiseContour:
    """Class to represent the visualiation of a contour"""

    def __init__(self, image, name, color=None, linewidth=2, linestyle="solid"):
        self.image = image
        self.name = name
        self.color = color
        self.linewidth = linewidth
        self.linestyle = linestyle


class VisualiseScalarOverlay:
    """Class to represent the visualiation of a scalar overlay"""

    def __init__(
        self,
        image,
        name,
        colormap=plt.cm.get_cmap("Spectral"),
        alpha=0.75,
        min_value=False,
        max_value=False,
        discrete_levels=False,
        mid_ticks=False,
        show_colorbar=True,
        norm=None,
        projection=False,
    ):
        self.image = image
        self.name = name
        self.colormap = colormap
        self.alpha = alpha
        self.min_value = min_value
        self.max_value = max_value
        self.discrete_levels = discrete_levels
        self.mid_ticks = mid_ticks
        self.show_colorbar = show_colorbar
        self.norm = norm
        self.projection = projection


class VisualiseVectorOverlay:
    """Class to represent the visualiation of a vector overlay"""

    def __init__(
        self,
        image,
        min_value=False,
        max_value=False,
        colormap=plt.cm.get_cmap("Spectral"),
        alpha=0.75,
        arrow_scale=0.25,
        arrow_width=1,
        subsample=4,
        color_function="perpendicular",
        invert_field=True,
        show_colorbar=True,
        name="input",
    ):
        self.image = image
        self.min_value = min_value
        self.max_value = max_value
        self.colormap = colormap
        self.alpha = alpha
        self.arrow_scale = arrow_scale
        self.arrow_width = arrow_width
        self.subsample = subsample
        self.color_function = color_function
        self.invert_field = invert_field
        self.show_colorbar = show_colorbar
        self.name = name


class VisualiseComparisonOverlay:
    """Class to represent the visualiation of a comparison image"""

    def __init__(self, image, name, color_rotation=0.35):
        self.image = image
        self.name = name
        self.color_rotation = color_rotation


class VisualiseBoundingBox:
    """Class to represent the visualiation of a bounding box"""

    def __init__(self, bounding_box, name, color="r", linewidth=2):

        if isinstance(bounding_box, sitk.Image):
            bounding_box = label_to_roi(bounding_box, return_as_list=True)

        self.bounding_box = bounding_box
        self.name = name
        self.color = color
        self.linewidth = linewidth


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


def generate_comparison_colormix(
    image_list, arr_slice=None, window=(-250, 500), color_rotation=0.35
):

    #! TO DO - make this function take in more than two images
    # Will need to use polar coordinates for HSV colorspace addition

    if len(image_list) == 2:
        if all([isinstance(image, sitk.Image) for image in image_list]):
            if any([image.GetDimension() >= 3 for image in image_list]) and arr_slice is None:
                raise ValueError("Images cannot be 3D unless 'arr_slice' is specified.")

            else:
                array_list = [
                    sitk.GetArrayViewFromImage(image).__getitem__(arr_slice)
                    for image in image_list
                ]

        elif all([isinstance(image, np.ndarray) for image in image_list]):
            if any([len(image.shape) >= 3 for image in image_list]) and arr_slice is None:
                raise ValueError("Images cannot be 3D unless 'arr_slice' is specified.")

            else:
                array_list = [image.__getitem__(arr_slice) for image in image_list]

    else:
        raise ValueError("'image_list' must be a list of two sitk.Image or np.ndarray.")

    nda_a, nda_b = array_list

    nda_a_norm = (np.clip(nda_a, window[0], window[0] + window[1]) - window[0]) / (window[1])
    nda_b_norm = (np.clip(nda_b, window[0], window[0] + window[1]) - window[0]) / (window[1])

    nda_color = np.stack(
        [
            color_rotation * (nda_a_norm > nda_b_norm)
            + (0.5 + color_rotation) * (nda_a_norm <= nda_b_norm),
            np.abs(nda_a_norm - nda_b_norm),
            (nda_a_norm + nda_b_norm) / 2,
        ],
        axis=-1,
    )

    return hsv2rgb(nda_color)


def project_onto_arbitrary_plane(
    image,
    projection_name="mean",
    projection_axis=0,
    rotation_axis=[1, 0, 0],
    rotation_angle=0,
    default_value=-1000,
    resample_interpolation=2,
):
    """[summary]

    Args:
        image ([type]): [description]
        projection_name (str, optional): [description]. Defaults to "mean".
        projection_axis (int, optional): [description]. Defaults to 0.
        rotation_axis (list, optional): [description]. Defaults to [1, 0, 0].
        rotation_angle (int, optional): [description]. Defaults to 0.
        default_value (int, optional): [description]. Defaults to -1000.
        resample_interpolation (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """

    projection_dict = {
        "sum": sitk.SumProjection,
        "mean": sitk.MeanProjection,
        "median": sitk.MedianProjection,
        "std": sitk.StandardDeviationProjection,
        "min": sitk.MinimumProjection,
        "max": sitk.MaximumProjection,
    }
    projection_function = projection_dict[projection_name]

    # Set centre as image centre
    rotation_centre = image.TransformContinuousIndexToPhysicalPoint(
        [(index - 1) / 2.0 for index in image.GetSize()]
    )

    # Define the transform, using predefined centre of rotation and given angle
    rotation_transform = sitk.VersorRigid3DTransform()
    rotation_transform.SetCenter(rotation_centre)
    rotation_transform.SetRotation(rotation_axis, rotation_angle)

    # Resample the image using the rotation transform
    resampled_image = sitk.Resample(
        image,
        rotation_transform,
        resample_interpolation,
        default_value,
        image.GetPixelID(),
    )

    # Project onto the given axis
    proj_image = projection_function(resampled_image, projection_axis)

    # Return this view
    image_slice = {
        0: proj_image[0, :, :],
        1: proj_image[:, 0, :],
        2: proj_image[:, :, 0],
    }

    return image_slice[projection_axis]
