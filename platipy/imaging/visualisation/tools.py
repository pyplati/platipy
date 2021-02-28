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

import pathlib

import math

import numpy as np
import SimpleITK as sitk

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


"""
This Python script comprises two contributions to the code base:
1) A bunch of helpful visualisation "helper" functions

2) A visualisation class used to generate figures of images, contours, vector fields, and more!
"""


def project_onto_arbitrary_plane(
    image,
    projection_name="mean",
    projection_axis=0,
    rotation_axis=[1, 0, 0],
    rotation_angle=0,
    default_value=-1000,
    resample_interpolation=2,
):

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


def generate_animation_from_image_sequence(
    image_list,
    output_file="animation.gif",
    fps=10,
    contour_list=False,
    scalar_list=False,
    figure_size_in=6,
    image_cmap=plt.cm.Greys_r,
    contour_cmap=plt.cm.jet,
    scalar_cmap=plt.cm.magma,
    image_window=[-1000, 800],
    scalar_min=False,
    scalar_max=False,
    scalar_alpha=0.5,
    image_origin="lower",
):

    # We need to check for ImageMagick
    # There may be other tools that can be used
    rcParams["animation.convert_path"] = r"/usr/bin/convert"
    convert_path = pathlib.Path(rcParams["animation.convert_path"])

    if not convert_path.exists():
        raise RuntimeError("To use this function you need ImageMagick.")

    if type(image_list[0]) is not sitk.Image:
        raise ValueError("Each image must be a SimplITK image (sitk.Image).")

    # Get the image information
    x_size, y_size = image_list[0].GetSize()
    x_spacing, y_spacing = image_list[1].GetSpacing()

    asp = y_spacing / x_spacing

    # Define the figure
    figure_size = (figure_size_in, figure_size_in * (asp * y_size) / (1.0 * x_size))
    fig, ax = plt.subplots(1, 1, figsize=(figure_size))

    # Display the first image
    # This will be updated
    display_image = ax.imshow(
        sitk.GetArrayFromImage(image_list[0]),
        aspect=asp,
        interpolation=None,
        origin=image_origin,
        cmap=image_cmap,
        clim=(image_window[0], image_window[0] + image_window[1]),
    )

    # We now deal with the contours
    # These can be given as a list of sitk.Image objects or a list of dicts {"name":sitk.Image}
    if contour_list is not False:

        if type(contour_list[0]) is not dict:
            plot_dict = {"_": contour_list[0]}
        else:
            plot_dict = contour_list[0]

        color_map = contour_cmap(np.linspace(0, 1, len(plot_dict)))

        for index, contour in enumerate(plot_dict.values()):

            display_contours = ax.contour(
                sitk.GetArrayFromImage(contour),
                colors=[color_map[index]],
                levels=[0],
                linewidths=2,
            )

    if scalar_list is not False:

        if scalar_min is False:
            scalar_min = np.min([sitk.GetArrayFromImage(i) for i in scalar_list])
        if scalar_max is False:
            scalar_max = np.max([sitk.GetArrayFromImage(i) for i in scalar_list])

        display_scalar = ax.imshow(
            np.ma.masked_outside(
                sitk.GetArrayFromImage(scalar_list[0]), scalar_min, scalar_max
            ),
            aspect=asp,
            interpolation=None,
            origin=image_origin,
            cmap=scalar_cmap,
            clim=(scalar_min, scalar_max),
            alpha=scalar_alpha,
            vmin=scalar_min,
            vmax=scalar_max,
        )

    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # The animate function does (you guessed it) the animation
    def animate(i):

        # Update the imaging data
        nda = sitk.GetArrayFromImage(image_list[i])
        display_image.set_data(nda)

        # TO DO - add in code for scalar overlay
        if contour_list is not False:
            try:
                ax.collections = []
            except ValueError:
                pass

            if type(contour_list[i]) is not dict:
                plot_dict = {"_": contour_list[i]}
            else:
                plot_dict = contour_list[i]

            color_map = contour_cmap(np.linspace(0, 1, len(plot_dict)))

            for index, contour in enumerate(plot_dict.values()):

                display_contours = ax.contour(
                    sitk.GetArrayFromImage(contour),
                    colors=[color_map[index]],
                    levels=[0],
                    linewidths=2,
                )

        if scalar_list is not False:
            nda = sitk.GetArrayFromImage(scalar_list[i])
            display_scalar.set_data(np.ma.masked_outside(nda, scalar_min, scalar_max))

        return (display_image,)

    # create animation using the animate() function with no repeat
    myAnimation = animation.FuncAnimation(
        fig,
        animate,
        frames=np.arange(0, len(image_list), 1),
        interval=10,
        blit=True,
        repeat=False,
    )

    # save animation at 30 frames per second
    myAnimation.save(output_file, writer="imagemagick", fps=fps)

    return myAnimation


class VisualiseContour:
    """Class to represent the visualiation of a contour"""

    def __init__(self, image, name, color=None, linewidth=2):
        self.image = image
        self.name = name
        self.color = color
        self.linewidth = linewidth


class VisualiseScalarOverlay:
    """Class to represent the visualiation of a scalar overlay"""

    def __init__(
        self,
        image,
        name,
        colormap=plt.cm.get_cmap("Spectral"),
        alpha=0.75,
        min_value=0.1,
        max_value=False,
        discrete_levels=False,
        mid_ticks=False,
        show_colorbar=True,
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


class VisualiseVectorOverlay:
    """Class to represent the visualiation of a vector overlay"""

    def __init__(
        self,
        image,
        name,
        colormap=plt.cm.get_cmap("Spectral"),
        alpha=0.75,
        arrow_scale=0.25,
        arrow_width=1,
        subsample=4,
        color_function="perpendicular",
        invert_field=True,
    ):
        self.image = image
        self.name = name
        self.colormap = colormap
        self.alpha = alpha
        self.arrow_scale = arrow_scale
        self.arrow_width = arrow_width
        self.subsample = subsample
        self.color_function = color_function
        self.invert_field = invert_field


class VisualiseComparisonOverlay:
    """Class to represent the visualiation of a comparison image"""

    def __init__(self, image, name, color_rotation=0.35):
        self.image = image
        self.color_rotation = color_rotation


class VisualiseBoundingBox:
    """Class to represent the visualiation of a bounding box"""

    def __init__(self, bounding_box, name, color=None):
        self.bounding_box = bounding_box
        self.name = name
        self.color = color


class ImageVisualiser:
    """Class to assist with visualising images and overlaying contours, scalars and bounding boxes."""

    def __init__(
        self,
        image,
        cut=None,
        axis="ortho",
        window=[-250, 500],
        figure_size_in=10,
        limits=None,
        colormap=plt.cm.Greys_r,
        origin="normal",
    ):
        self.__set_image(image)
        self.__contours = []
        self.__bounding_boxes = []
        self.__scalar_overlays = []
        self.__vector_overlays = []
        self.__comparison_overlays = []
        self.__show_legend = False
        self.__show_colorbar = False
        self.__figure = None
        self.__figure_size = figure_size_in
        self.__window = window
        self.__axis = axis
        self.__cut = cut
        self.__limits = limits
        self.__colormap = colormap
        self.__origin = origin

        self.clear()

    def __set_image(self, image):
        self.__image = image

    def __set_labelmap(self, labelmap, labels=None):

        # TODO: Convert label map to binary masks for display

        raise NotImplementedError

    image = property(fset=__set_image)
    # contours = property(fset=__set_contours)
    labelmap = property(fset=__set_labelmap)

    def clear(self):
        """Clear all overlays"""

        self.__contours = []
        self.__bounding_boxes = []
        self.__scalar_overlays = []
        self.__comparison_overlays = []
        self.__vector_overlays = []

    def set_limits_from_label(self, label, expansion=[0, 0, 0], min_value=0):

        label_stats_image_filter = sitk.LabelStatisticsImageFilter()
        label_stats_image_filter.Execute(label, label > 0)
        bounding_box = np.array(label_stats_image_filter.GetBoundingBox(1))

        index = [bounding_box[x * 2] for x in range(3)]
        size = [bounding_box[(x * 2) + 1] - bounding_box[x * 2] + 1 for x in range(3)]

        if hasattr(expansion, "__iter__"):
            expansion = np.array(expansion) / np.array(label.GetSpacing()[::-1])

        else:
            expansion = np.repeat(expansion, 3) / np.array(label.GetSpacing()[::-1])

        expansion = np.array(expansion[::-1])

        # Avoid starting outside the image
        sag_0, cor_0, ax_0 = np.max([index - expansion, np.array([0, 0, 0])], axis=0)

        # Avoid ending outside the image
        sag_size, cor_size, ax_size = np.min(
            [
                np.array(label.GetSize()) - np.array([sag_0, cor_0, ax_0]),
                np.array(size) + 2 * expansion,
            ],
            axis=0,
        )

        # ax_0, cor_0, sag_0 = ax_0-

        if self.__axis == "ortho":
            self.__limits = [
                ax_0,
                ax_0 + ax_size,
                cor_0,
                cor_0 + cor_size,
                sag_0,
                sag_0 + sag_size,
            ]

        if self.__axis == "x":
            self.__limits = [cor_0, cor_0 + cor_size, ax_0, ax_0 + ax_size]
        if self.__axis == "y":
            self.__limits = [sag_0, sag_0 + sag_size, ax_0, ax_0 + ax_size]
        if self.__axis == "z":
            self.__limits = [sag_0, sag_0 + sag_size, cor_0, cor_0 + cor_size]

    def add_contour(
        self, contour, name=None, color=None, colorbase=plt.cm.rainbow, linewidth=2
    ):
        """Add a contour as overlay

        Args:
            contour (sitk.Image|dict): Contour mask or dict containing contour masks.
            name (str, optional): Name to give the contour (only used if passing sitk.Image as
                                  contour). Defaults to None.
            color (str|tuple|list, optional): The color to use when drawing the contour(s).
                                              Defaults to None.

        Raises:
            ValueError: Contour must be dict of sitk.Image.
            ValueError: If passing a dict for contour, all values must be sitk.Image.
        """

        if isinstance(contour, dict):

            self.__show_legend = True

            if not all(map(lambda i: isinstance(i, sitk.Image), contour.values())):
                raise ValueError(
                    "When passing dict, all values must be of type SimpleITK.Image"
                )

            for contour_name in contour:

                if isinstance(color, dict):
                    try:
                        contour_color = color[contour_name]
                    except:
                        contour_color = None
                else:
                    contour_color = color

                visualise_contour = VisualiseContour(
                    contour[contour_name],
                    contour_name,
                    color=contour_color,
                    linewidth=linewidth,
                )
                self.__contours.append(visualise_contour)

        elif isinstance(contour, sitk.Image):

            # Use a default name if not specified
            if not name:
                name = "input"
                self.__show_legend = False

            visualise_contour = VisualiseContour(
                contour, name, color=color, linewidth=linewidth
            )
            self.__contours.append(visualise_contour)
        else:

            raise ValueError(
                "Contours should be represented as a dict with contour name as key "
                "and sitk.Image as value, or as an sitk.Image and passing the contour_name"
            )

        self.__contour_color_base = colorbase

    def add_scalar_overlay(
        self,
        scalar_image,
        name=None,
        colormap=plt.cm.get_cmap("Spectral"),
        alpha=0.75,
        min_value=0.1,
        max_value=False,
        discrete_levels=False,
        mid_ticks=False,
        show_colorbar=True,
    ):
        """Overlay a scalar image on to the existing image

        Args:
            scalar_image sitk.Image|dict): Scalar image or dict containing scalar images.
            name (str, optional): Name to give the scalar image (only used if passing sitk.Image as
                                  scalar image). Defaults to None.
            colormap (matplotlib.colors.Colormap, optional): The colormap to be used when
                                                             overlaying this scalar image. Defaults
                                                             to plt.cm.get_cmap("Spectral").
            alpha (float, optional): Alpha to apply to overlay. Defaults to 0.75.
            min_value (float, optional): Values below this value aren't rendered. Defaults to 0.1.

        Raises:
            ValueError: Scalar overlay must be dict of sitk.Image.
            ValueError: If passing a dict for contour, all values must be sitk.Image.
        """

        self.__show_colorbar = True

        if isinstance(scalar_image, dict):

            if not all(map(lambda i: isinstance(i, sitk.Image), scalar_image.values())):
                raise ValueError(
                    "When passing dict, all values must be of type SimpleITK.Image"
                )

            for name in scalar_image:
                visualise_contour = VisualiseScalarOverlay(
                    scalar_image[name],
                    name,
                    colormap=colormap,
                    alpha=alpha,
                    min_value=min_value,
                    max_value=max_value,
                    discrete_levels=discrete_levels,
                    mid_ticks=mid_ticks,
                    show_colorbar=show_colorbar,
                )
                self.__scalar_overlays.append(visualise_contour)

        elif isinstance(scalar_image, sitk.Image):

            # Use a default name if not specified
            if not name:
                name = "input"
                self.__show_legend = False

            visualise_contour = VisualiseScalarOverlay(
                scalar_image,
                name,
                colormap=colormap,
                alpha=alpha,
                min_value=min_value,
                max_value=max_value,
                discrete_levels=discrete_levels,
                mid_ticks=mid_ticks,
                show_colorbar=show_colorbar,
            )
            self.__scalar_overlays.append(visualise_contour)
        else:

            raise ValueError(
                "Contours should be represented as a dict with contour name as key "
                "and sitk.Image as value, or as an sitk.Image and passing the contour_name"
            )

    def add_vector_overlay(
        self,
        vector_image,
        name=None,
        colormap=plt.cm.get_cmap("Spectral"),
        alpha=0.75,
        arrow_scale=0.25,
        arrow_width=1,
        subsample=4,
        color_function="perpendicular",
    ):
        """Overlay a vector field on to the existing image

        Args:
            vector_image sitk.Image|dict): Vector image (will be displayed as ).
            name (str, optional): Name to give the vector field (only used if passing
                                  sitk.Image as vector field). Defaults to None.
            colormap (matplotlib.colors.Colormap, optional): The colormap to be used when
                                                             overlaying this vector field. Defaults
                                                             to plt.cm.get_cmap("Spectral").
            alpha (float, optional): Alpha to apply to overlay vectors. Defaults to 0.75.
            arrow_scale (float, optional): Relative scaling of vectors. Defaults to 0.25.
            arrow_width (float, optional): Width of vector field arrow. Defaults to 0.25.
            subsample (int, optional): Defines to subsampling ratio of displayed vectors.
                                       Defaults to 4.
            color_function (str, optional): Determines how vectors are colored. Options:
                                            'perpendicular' - vectors colored by perpendicular value
                                            'magnitude' - vectors colored by magnitude.

        Raises:
            ValueError: Vector overlay must be of type sitk.Image.
        """

        if (
            isinstance(vector_image, sitk.Image)
            and vector_image.GetNumberOfComponentsPerPixel() > 1
        ):

            # Use a default name if not specified
            if not name:
                name = "input"
                self.__show_legend = False

            visualise_vector_field = VisualiseVectorOverlay(
                vector_image,
                name,
                colormap=colormap,
                alpha=alpha,
                arrow_scale=arrow_scale,
                arrow_width=arrow_width,
                subsample=subsample,
                color_function=color_function,
            )
            self.__vector_overlays.append(visualise_vector_field)
        else:

            raise ValueError("Vector field should be sitk.Image (of vector type).")

    def add_comparison_overlay(self, image, name=None, color_rotation=0.35):
        """Overlay a comparison image on the existing image

        Args:
            image sitk.Image): Image (will be displayed as a comparison).
            name (str, optional): Name to give the image. Defaults to None.
            color_rotation (float, optional): Defines the hue of the original image (0 - 0.5).

        Raises:
            ValueError: Comparison overlay must be of type sitk.Image.
        """

        if isinstance(image, sitk.Image):

            # Use a default name if not specified
            if not name:
                name = "input"
                self.__show_legend = False

            visualise_comparison = VisualiseComparisonOverlay(
                image, name, color_rotation=color_rotation
            )
            self.__comparison_overlays.append(visualise_comparison)
        else:

            raise ValueError("Image should be sitk.Image.")

    def add_bounding_box(self, bounding_box, name=None, color=None):

        self.__show_legend = True

        if isinstance(bounding_box, dict):

            if not all(
                map(
                    lambda i: isinstance(i, (list, tuple)) and len(i) == 6,
                    bounding_box.values(),
                )
            ):
                raise ValueError(
                    "All values must be of type list or tuple with length 6"
                )

            for name in bounding_box:
                visualise_bounding_box = VisualiseBoundingBox(
                    bounding_box[name], name, color=color
                )
                self.__bounding_boxes.append(visualise_bounding_box)

        elif isinstance(bounding_box, (list, tuple)):

            # Use a default name if not specified
            if not name:
                name = "input"
                self.__show_legend = False

            visualise_bounding_box = VisualiseBoundingBox(
                bounding_box, name, color=color
            )
            self.__bounding_boxes.append(visualise_bounding_box)

        else:
            raise ValueError(
                "Bounding boxes should be represented as a dict with bounding box name as key "
                "and list or tuple as value"
            )

    def return_slice(self, axis, index):
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

    def subsample_vector_field(self, axis, cut, subsample=1):
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

    def vector_image_grid(self, axis, vector_field_array, subsample=1):
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

    def reorientate_vector_field(
        self, axis, vector_ax, vector_cor, vector_sag, invert_field=True
    ):
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

    def show(self, interact=False):
        """Render the image with all overlays"""
        if len(self.__comparison_overlays) == 0:
            self.display_slice()
        else:
            self.overlay_comparison()

        self.overlay_scalar_field()
        self.overlay_vector_field()
        self.overlay_contours()
        self.overlay_bounding_boxes()

        self.adjust_view()

        if interact:
            self.interact_adjust_slice()

        return self.__figure

    def interact_adjust_slice(self):
        image = self.__image
        nda = sitk.GetArrayViewFromImage(image)
        (ax_size, cor_size, sag_size) = nda.shape[:3]

        image_view = self.__image_view

        # ~10x speed-up by pre-contructing views
        arr_slices_ax = {
            i: nda.__getitem__(self.return_slice("z", i)) for i in range(ax_size)
        }
        arr_slices_cor = {
            i: nda.__getitem__(self.return_slice("y", i)) for i in range(cor_size)
        }
        arr_slices_sag = {
            i: nda.__getitem__(self.return_slice("x", i)) for i in range(sag_size)
        }

        if self.__cut is None:
            slice_ax = int(ax_size / 2.0)
            slice_cor = int(cor_size / 2.0)
            slice_sag = int(sag_size / 2.0)

            self.__cut = [slice_ax, slice_cor, slice_sag]

        for view_name in image_view.keys():
            if view_name == "ax_view":

                ax_view = image_view["ax_view"]

                widget = widgets.IntSlider(
                    min=0, max=ax_size, step=1, value=self.__cut[0]
                )

                def f_adjust(axial_slice):
                    ax_view.set_data(arr_slices_ax[axial_slice])
                    return

                interact(f_adjust, axial_slice=widget)

            if view_name == "cor_view":

                cor_view = image_view["cor_view"]

                widget = widgets.IntSlider(
                    min=0, max=cor_size, step=1, value=self.__cut[1]
                )

                def f_adjust(coronal_slice):
                    cor_view.set_data(arr_slices_cor[coronal_slice])
                    return

                interact(f_adjust, coronal_slice=widget)

            if view_name == "sag_view":

                sag_view = image_view["sag_view"]

                widget = widgets.IntSlider(
                    min=0, max=sag_size, step=1, value=self.__cut[2]
                )

                def f_adjust(sagittal_slice):
                    sag_view.set_data(arr_slices_sag[sagittal_slice])
                    return

                interact(f_adjust, sagittal_slice=widget)

    def display_slice(self):
        """Display the configured image slice"""

        image = self.__image
        nda = sitk.GetArrayFromImage(image)

        (ax_size, cor_size, sag_size) = nda.shape[:3]

        try:
            rgb_flag = nda.shape[3] == 3
            print(
                "Found a (z,y,x,3) dimensional array - assuming this is an RGB image."
            )
            nda /= nda.max()
        except:
            None

        sp_plane, _, sp_slice = image.GetSpacing()
        asp = (1.0 * sp_slice) / sp_plane

        if self.__axis == "ortho":
            figure_size = (
                self.__figure_size,
                self.__figure_size
                * (asp * ax_size + cor_size)
                / (1.0 * sag_size + cor_size),
            )

            self.__figure, ((ax_ax, blank), (ax_cor, ax_sag)) = plt.subplots(
                2,
                2,
                figsize=figure_size,
                gridspec_kw={
                    "height_ratios": [(cor_size) / (asp * ax_size), 1],
                    "width_ratios": [sag_size, cor_size],
                },
            )
            blank.axis("off")

            if self.__cut is None:
                slice_ax = int(ax_size / 2.0)
                slice_cor = int(cor_size / 2.0)
                slice_sag = int(sag_size / 2.0)

                self.__cut = [slice_ax, slice_cor, slice_sag]

            s_ax = self.return_slice("z", self.__cut[0])
            s_cor = self.return_slice("y", self.__cut[1])
            s_sag = self.return_slice("x", self.__cut[2])

            ax_view = ax_ax.imshow(
                nda.__getitem__(s_ax),
                aspect=1.0,
                interpolation=None,
                origin={"normal": "upper", "reversed": "lower"}[self.__origin],
                cmap=self.__colormap,
                clim=(self.__window[0], self.__window[0] + self.__window[1]),
            )
            cor_view = ax_cor.imshow(
                nda.__getitem__(s_cor),
                origin="lower",
                aspect=asp,
                interpolation=None,
                cmap=self.__colormap,
                clim=(self.__window[0], self.__window[0] + self.__window[1]),
            )
            sag_view = ax_sag.imshow(
                nda.__getitem__(s_sag),
                origin="lower",
                aspect=asp,
                interpolation=None,
                cmap=self.__colormap,
                clim=(self.__window[0], self.__window[0] + self.__window[1]),
            )

            ax_ax.axis("off")
            ax_cor.axis("off")
            ax_sag.axis("off")

            self.__figure.subplots_adjust(
                left=0, right=1, wspace=0.01, hspace=0.01, top=1, bottom=0
            )

            self.__image_view = {
                "ax_view": ax_view,
                "cor_view": cor_view,
                "sag_view": sag_view,
            }

        else:

            if hasattr(self.__cut, "__iter__"):
                warnings.warn(
                    "You have selected a single axis and multiple slice locations, attempting to match."
                )
                self.__cut = self.__cut[{"x": 2, "y": 1, "z": 0}[self.__axis]]

            if self.__axis == "x" or self.__axis == "sag":
                axis_view_name_consistent = "sag_view"
                figure_size = (
                    self.__figure_size,
                    self.__figure_size * (asp * ax_size) / (1.0 * cor_size),
                )
                self.__figure, ax = plt.subplots(1, 1, figsize=(figure_size))
                org = "lower"
                if not self.__cut:
                    self.__cut = int(sag_size / 2.0)

            if self.__axis == "y" or self.__axis == "cor":
                axis_view_name_consistent = "cor_view"
                figure_size = (
                    self.__figure_size,
                    self.__figure_size * (asp * ax_size) / (1.0 * sag_size),
                )
                self.__figure, ax = plt.subplots(1, 1, figsize=(figure_size))
                org = "lower"
                if not self.__cut:
                    self.__cut = int(cor_size / 2.0)

            if self.__axis == "z" or self.__axis == "ax":
                axis_view_name_consistent = "ax_view"
                asp = 1
                figure_size = (
                    self.__figure_size,
                    self.__figure_size * (asp * cor_size) / (1.0 * sag_size),
                )
                self.__figure, ax = plt.subplots(1, 1, figsize=(figure_size))
                org = {"normal": "upper", "reversed": "lower"}[self.__origin]
                if not self.__cut:
                    self.__cut = int(ax_size / 2.0)

            s = self.return_slice(self.__axis, self.__cut)
            ax_indiv = ax.imshow(
                nda.__getitem__(s),
                aspect=asp,
                interpolation=None,
                origin=org,
                cmap=self.__colormap,
                clim=(self.__window[0], self.__window[0] + self.__window[1]),
            )
            ax.axis("off")

            self.__figure.subplots_adjust(left=0, right=1, bottom=0, top=1)

            self.__image_view = {axis_view_name_consistent: ax_indiv}

    def overlay_comparison(self):
        """Display an overlay comparison

        Args:
            color_rotation (float, optional): The hue used to color the original image (0 - 0.5).
        """

        if len(self.__comparison_overlays) > 1:
            raise ValueError("You can only display one comparison image.")

        else:
            comparison_overlay = self.__comparison_overlays[0]

        image_original = self.__image
        nda_original = sitk.GetArrayFromImage(image_original)

        image_new = comparison_overlay.image
        nda_new = sitk.GetArrayFromImage(image_new)
        color_rotation = comparison_overlay.color_rotation

        (ax_size, cor_size, sag_size) = nda_original.shape
        sp_plane, _, sp_slice = image_original.GetSpacing()
        asp = (1.0 * sp_slice) / sp_plane

        window = self.__window

        if self.__axis == "ortho":
            figure_size = (
                self.__figure_size,
                self.__figure_size
                * (asp * ax_size + cor_size)
                / (1.0 * sag_size + cor_size),
            )

            self.__figure, ((ax_ax, blank), (ax_cor, ax_sag)) = plt.subplots(
                2,
                2,
                figsize=figure_size,
                gridspec_kw={
                    "height_ratios": [(cor_size) / (asp * ax_size), 1],
                    "width_ratios": [sag_size, cor_size],
                },
            )
            blank.axis("off")

            if self.__cut is None:
                slice_ax = int(ax_size / 2.0)
                slice_cor = int(cor_size / 2.0)
                slice_sag = int(sag_size / 2.0)

                self.__cut = [slice_ax, slice_cor, slice_sag]

            s_ax = self.return_slice("z", self.__cut[0])
            s_cor = self.return_slice("y", self.__cut[1])
            s_sag = self.return_slice("x", self.__cut[2])

            nda_a = nda_original.__getitem__(s_ax)
            nda_b = nda_new.__getitem__(s_ax)

            nda_a_norm = (
                np.clip(nda_a, window[0], window[0] + window[1]) - window[0]
            ) / (window[1])
            nda_b_norm = (
                np.clip(nda_b, window[0], window[0] + window[1]) - window[0]
            ) / (window[1])

            nda_colour = np.stack(
                [
                    color_rotation * (nda_a_norm > nda_b_norm)
                    + (0.5 + color_rotation) * (nda_a_norm <= nda_b_norm),
                    np.abs(nda_a_norm - nda_b_norm),
                    (nda_a_norm + nda_b_norm) / 2,
                ],
                axis=-1,
            )

            ax_ax.imshow(
                hsv2rgb(nda_colour),
                aspect=1.0,
                origin={"normal": "upper", "reversed": "lower"}[self.__origin],
                interpolation=None,
            )

            nda_a = nda_original.__getitem__(s_cor)
            nda_b = nda_new.__getitem__(s_cor)

            nda_a_norm = (
                np.clip(nda_a, window[0], window[0] + window[1]) - window[0]
            ) / (window[1])
            nda_b_norm = (
                np.clip(nda_b, window[0], window[0] + window[1]) - window[0]
            ) / (window[1])

            nda_colour = np.stack(
                [
                    color_rotation * (nda_a_norm > nda_b_norm)
                    + (0.5 + color_rotation) * (nda_a_norm <= nda_b_norm),
                    np.abs(nda_a_norm - nda_b_norm),
                    (nda_a_norm + nda_b_norm) / 2,
                ],
                axis=-1,
            )

            ax_cor.imshow(
                hsv2rgb(nda_colour),
                origin="lower",
                aspect=asp,
                interpolation=None,
            )

            nda_a = nda_original.__getitem__(s_sag)
            nda_b = nda_new.__getitem__(s_sag)

            nda_a_norm = (
                np.clip(nda_a, window[0], window[0] + window[1]) - window[0]
            ) / (window[1])
            nda_b_norm = (
                np.clip(nda_b, window[0], window[0] + window[1]) - window[0]
            ) / (window[1])

            nda_colour = np.stack(
                [
                    color_rotation * (nda_a_norm > nda_b_norm)
                    + (0.5 + color_rotation) * (nda_a_norm <= nda_b_norm),
                    np.abs(nda_a_norm - nda_b_norm),
                    (nda_a_norm + nda_b_norm) / 2,
                ],
                axis=-1,
            )

            ax_sag.imshow(
                hsv2rgb(nda_colour),
                origin="lower",
                aspect=asp,
                interpolation=None,
            )

            ax_ax.axis("off")
            ax_cor.axis("off")
            ax_sag.axis("off")

            self.__figure.subplots_adjust(
                left=0, right=1, wspace=0.01, hspace=0.01, top=1, bottom=0
            )

        else:

            if hasattr(self.__cut, "__iter__"):
                warnings.warn(
                    "You have selected a single axis and multiple slice locations, attempting to match."
                )
                self.__cut = self.__cut[{"x": 2, "y": 1, "z": 0}[self.__axis]]

            if self.__axis == "x" or self.__axis == "sag":
                figure_size = (
                    self.__figure_size,
                    self.__figure_size * (asp * ax_size) / (1.0 * cor_size),
                )
                self.__figure, ax = plt.subplots(1, 1, figsize=(figure_size))
                org = "lower"
                if not self.__cut:
                    self.__cut = int(sag_size / 2.0)

            if self.__axis == "y" or self.__axis == "cor":
                figure_size = (
                    self.__figure_size,
                    self.__figure_size * (asp * ax_size) / (1.0 * sag_size),
                )
                self.__figure, ax = plt.subplots(1, 1, figsize=(figure_size))
                org = "lower"
                if not self.__cut:
                    self.__cut = int(cor_size / 2.0)

            if self.__axis == "z" or self.__axis == "ax":
                asp = 1
                figure_size = (
                    self.__figure_size,
                    self.__figure_size * (asp * cor_size) / (1.0 * sag_size),
                )
                self.__figure, ax = plt.subplots(1, 1, figsize=(figure_size))
                org = "upper"
                if not self.__cut:
                    self.__cut = int(ax_size / 2.0)

            s = self.return_slice(self.__axis, self.__cut)

            nda_a = nda_original.__getitem__(s)
            nda_b = nda_new.__getitem__(s)

            nda_a_norm = (
                np.clip(nda_a, window[0], window[0] + window[1]) - window[0]
            ) / (window[1])
            nda_b_norm = (
                np.clip(nda_b, window[0], window[0] + window[1]) - window[0]
            ) / (window[1])

            nda_colour = np.stack(
                [
                    color_rotation * (nda_a_norm > nda_b_norm)
                    + (0.5 + color_rotation) * (nda_a_norm <= nda_b_norm),
                    np.abs(nda_a_norm - nda_b_norm),
                    (nda_a_norm + nda_b_norm) / 2,
                ],
                axis=-1,
            )

            ax.imshow(
                hsv2rgb(nda_colour),
                aspect=asp,
                interpolation=None,
                origin=org,
            )
            ax.axis("off")

            self.__figure.subplots_adjust(left=0, right=1, bottom=0, top=1)

    def adjust_view(self):
        """adjust_view is a helper function for modifying axis limits.
        Specify *limits* when initialising the ImageVisulaiser to use.
        Alternatively, use set_limits_from_label to specify automatically.
        """

        limits = self.__limits
        origin = self.__origin

        if limits is not None:
            if self.__axis == "ortho":
                ax_ax, _, ax_cor, ax_sag = self.__figure.axes[:4]

                if len(self.__figure.axes) == 5:
                    cax = self.__figure.axes[4]
                else:
                    cax = False

                ax_orig_0, ax_orig_1 = ax_cor.get_ylim()
                cor_orig_0, cor_orig_1 = ax_ax.get_ylim()
                sag_orig_0, sag_orig_1 = ax_ax.get_xlim()

                ax_0, ax_1, cor_0, cor_1, sag_0, sag_1 = limits

                # Perform some corrections
                ax_0, ax_1 = sorted([ax_0, ax_1])
                cor_0, cor_1 = sorted([cor_0, cor_1])
                sag_0, sag_1 = sorted([sag_0, sag_1])

                ax_orig_0, ax_orig_1 = sorted([ax_orig_0, ax_orig_1])
                cor_orig_0, cor_orig_1 = sorted([cor_orig_0, cor_orig_1])
                sag_orig_0, sag_orig_1 = sorted([sag_orig_0, sag_orig_1])

                ax_size = ax_1 - ax_0
                cor_size = cor_1 - cor_0
                sag_size = sag_1 - sag_0

                asp = ax_cor.get_aspect()

                ratio_x = ((cor_1 - cor_0) + (sag_1 - sag_0)) / (
                    (cor_orig_1 - cor_orig_0) + (sag_orig_1 - sag_orig_0)
                )
                ratio_y = (1 / asp * (cor_1 - cor_0) + (ax_1 - ax_0)) / (
                    1 / asp * (cor_orig_1 - cor_orig_0) + (ax_orig_1 - ax_orig_0)
                )

                if origin is "reversed":
                    cor_0, cor_1 = cor_1, cor_0

                ax_ax.set_xlim(sag_0, sag_1)
                ax_ax.set_ylim(cor_1, cor_0)

                ax_cor.set_xlim(sag_0, sag_1)
                ax_cor.set_ylim(ax_0, ax_1)

                ax_sag.set_xlim(cor_0, cor_1)
                ax_sag.set_ylim(ax_0, ax_1)

                gs = gridspec.GridSpec(
                    2,
                    2,
                    height_ratios=[(cor_size) / (asp * ax_size), 1],
                    width_ratios=[sag_size, cor_size],
                )

                ax_ax.set_position(gs[0].get_position(self.__figure))
                ax_ax.set_subplotspec(gs[0])

                ax_cor.set_position(gs[2].get_position(self.__figure))
                ax_cor.set_subplotspec(gs[2])

                ax_sag.set_position(gs[3].get_position(self.__figure))
                ax_sag.set_subplotspec(gs[3])

                fig_size_x, fig_size_y = self.__figure.get_size_inches()
                fig_size_y = fig_size_y * ratio_y / ratio_x

                ax_ax_bbox = gs[0].get_position(self.__figure)

                if cax is not False:
                    cax.set_position(
                        (
                            ax_ax_bbox.x1 + 0.02,
                            ax_ax_bbox.y0 + 0.01,
                            0.05,
                            ax_ax_bbox.height - 0.02,
                        )
                    )

                self.__figure.set_size_inches(fig_size_x, fig_size_y)

            elif self.__axis in ["x", "y", "z"]:
                ax = self.__figure.axes[0]
                x_orig_0, x_orig_1 = ax.get_xlim()
                y_orig_0, y_orig_1 = ax.get_ylim()

                x_0, x_1, y_0, y_1 = limits
                # Perform some corrections
                x_0, x_1 = sorted([x_0, x_1])
                y_0, y_1 = sorted([y_0, y_1])

                if self.__axis == "z":
                    y_0, y_1 = y_1, y_0

                ratio_x = np.abs(x_1 - x_0) / np.abs(x_orig_1 - x_orig_0)
                ratio_y = np.abs(y_1 - y_0) / np.abs(y_orig_1 - y_orig_0)

                ax.set_xlim(x_0, x_1)
                ax.set_ylim(y_0, y_1)

                fig_size_x, fig_size_y = self.__figure.get_size_inches()
                fig_size_y = fig_size_y * ratio_y / ratio_x

                self.__figure.set_size_inches(fig_size_x, fig_size_y)

    def overlay_contours(self):
        """Overlay the contours on to the current figure image"""

        if len(self.__contours) == 0:
            return

        plot_dict = {}
        color_dict = {}

        color_gen_index = 0

        for contour in self.__contours:
            contour_image_resampled = sitk.Resample(contour.image, self.__image)
            plot_dict[contour.name] = sitk.GetArrayFromImage(contour_image_resampled)

            if contour.color is not None:
                color_dict[contour.name] = contour.color
            else:
                color_map = self.__contour_color_base(
                    np.linspace(0, 1, len(self.__contours))
                )

                color_dict[contour.name] = color_map[color_gen_index % 255]
                color_gen_index += 1

        linewidths = [contour.linewidth for contour in self.__contours]

        # Test types of axes
        axes = self.__figure.axes[:4]

        if self.__axis in ["x", "y", "z"]:
            ax = axes[0]
            s = self.return_slice(self.__axis, self.__cut)

            for index, c_name in enumerate(plot_dict.keys()):
                try:
                    ax.contour(
                        plot_dict[c_name].__getitem__(s),
                        colors=[color_dict[c_name]],
                        levels=[0],
                        # alpha=0.8,
                        linewidths=linewidths,
                        label=c_name,
                        origin="lower",
                    )
                except:
                    pass

        elif self.__axis == "ortho":
            ax_ax, _, ax_cor, ax_sag = axes

            ax = ax_ax

            s_ax = self.return_slice("z", self.__cut[0])
            s_cor = self.return_slice("y", self.__cut[1])
            s_sag = self.return_slice("x", self.__cut[2])

            for index, c_name in enumerate(plot_dict.keys()):

                temp = ax_ax.contour(
                    plot_dict[c_name].__getitem__(s_ax),
                    levels=[0],
                    linewidths=linewidths,
                    colors=[color_dict[c_name]],
                    origin="lower",
                )
                temp.collections[0].set_label(c_name)

                ax_cor.contour(
                    plot_dict[c_name].__getitem__(s_cor),
                    levels=[0],
                    linewidths=linewidths,
                    colors=[color_dict[c_name]],
                    origin="lower",
                )
                ax_sag.contour(
                    plot_dict[c_name].__getitem__(s_sag),
                    levels=[0],
                    linewidths=linewidths,
                    colors=[color_dict[c_name]],
                    origin="lower",
                )
            if len(self.__figure.axes) == 5:
                pad = 1.35
            else:
                pad = 1.05
            if self.__show_legend:
                approx_scaling = self.__figure_size / (len(plot_dict.keys()))
                ax.legend(
                    loc="center left",
                    bbox_to_anchor=(pad, 0.5),
                    fontsize=min([10, 16 * approx_scaling]),
                )

        else:
            raise ValueError('Axis is must be one of "x","y","z","ortho".')

    def overlay_scalar_field(self):
        """Overlay the scalar image onto the existing figure"""

        for scalar in self.__scalar_overlays:

            scalar_image = scalar.image
            nda = sitk.GetArrayFromImage(scalar_image)

            alpha = scalar.alpha
            sMin = scalar.min_value

            if scalar.max_value:
                sMax = scalar.max_value
            else:
                sMax = nda.max()

            if scalar.discrete_levels:
                colormap_name = scalar.colormap.name
                colormap = plt.cm.get_cmap(colormap_name, scalar.discrete_levels)

            else:
                colormap = scalar.colormap

            # nda = nda / nda.max()
            nda = np.ma.masked_less_equal(nda, sMin)

            sp_plane, _, sp_slice = scalar_image.GetSpacing()
            asp = (1.0 * sp_slice) / sp_plane

            # Test types of axes
            axes = self.__figure.axes[:4]
            if len(axes) < 4:
                ax = axes[0]
                s = self.return_slice(self.__axis, self.__cut)
                if self.__axis == "z":
                    org = {"normal": "upper", "reversed": "lower"}[self.__origin]
                else:
                    org = "lower"
                sp = ax.imshow(
                    nda.__getitem__(s),
                    interpolation=None,
                    cmap=colormap,
                    clim=(sMin, sMax),
                    aspect={"z": 1, "y": asp, "x": asp}[self.__axis],
                    origin=org,
                    vmin=sMin,
                    vmax=sMax,
                    alpha=alpha,
                )

                if scalar.show_colorbar:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = self.__figure.colorbar(sp, cax=cax, orientation="vertical")
                    cbar.set_label(scalar.name)
                    cbar.solids.set_alpha(1)
                    if scalar.discrete_levels:
                        cbar.set_ticks(
                            np.linspace(sMin, sMax, scalar.discrete_levels + 1)
                        )

                    fX, fY = self.__figure.get_size_inches()
                    self.__figure.set_size_inches(fX * 1.15, fY)
                    self.__figure.subplots_adjust(left=0, right=0.88, bottom=0, top=1)

            elif len(axes) == 4:
                ax_ax, _, ax_cor, ax_sag = axes

                sAx = self.return_slice("z", self.__cut[0])
                sCor = self.return_slice("y", self.__cut[1])
                sSag = self.return_slice("x", self.__cut[2])

                sp = ax_ax.imshow(
                    nda.__getitem__(sAx),
                    interpolation=None,
                    cmap=colormap,
                    clim=(sMin, sMax),
                    aspect=1,
                    origin={"normal": "upper", "reversed": "lower"}[self.__origin],
                    vmin=sMin,
                    vmax=sMax,
                    alpha=alpha,
                )

                ax_cor.imshow(
                    nda.__getitem__(sCor),
                    interpolation=None,
                    cmap=colormap,
                    clim=(sMin, sMax),
                    origin="lower",
                    aspect=asp,
                    vmin=sMin,
                    vmax=sMax,
                    alpha=alpha,
                )

                ax_sag.imshow(
                    nda.__getitem__(sSag),
                    interpolation=None,
                    cmap=colormap,
                    clim=(sMin, sMax),
                    origin="lower",
                    aspect=asp,
                    vmin=sMin,
                    vmax=sMax,
                    alpha=alpha,
                )

                if scalar.show_colorbar:

                    # divider = make_axes_locatable(ax_ax)
                    # cax = divider.append_axes("right", size="5%", pad=0.05)

                    ax_box = ax_ax.get_position(original=False)
                    cbar_width = ax_box.width * 0.05  # 5% of axis width

                    cax = self.__figure.add_axes(
                        (
                            ax_box.x1 + 0.02,
                            ax_box.y0 + 0.01,
                            cbar_width,
                            ax_box.height - 0.02,
                        )
                    )

                    cbar = self.__figure.colorbar(sp, cax=cax, orientation="vertical")
                    cbar.set_label(scalar.name)
                    cbar.solids.set_alpha(1)
                    if scalar.discrete_levels:

                        if scalar.mid_ticks:

                            delta_tick = (sMax - sMin) / scalar.discrete_levels
                            cbar.set_ticks(
                                np.linspace(
                                    sMin + delta_tick / 2,
                                    sMax - delta_tick / 2,
                                    scalar.discrete_levels,
                                )
                            )

                        else:
                            cbar.set_ticks(
                                np.linspace(
                                    sMin,
                                    sMax,
                                    scalar.discrete_levels + 1,
                                )
                            )

    def overlay_vector_field(self):
        """Overlay vector field onto existing figure"""
        for vector in self.__vector_overlays:

            image = vector.image
            name = vector.name
            colormap = vector.colormap
            alpha = vector.alpha
            arrow_scale = vector.arrow_scale
            arrow_width = vector.arrow_width
            subsample = vector.subsample
            color_function = vector.color_function
            invert_field = vector.invert_field

            inverse_vector_image = image  # sitk.InvertDisplacementField(image)
            vector_nda = sitk.GetArrayFromImage(inverse_vector_image)

            sp_plane, _, sp_slice = image.GetSpacing()
            asp = (1.0 * sp_slice) / sp_plane

            # Test types of axes
            axes = self.__figure.axes[:4]
            if len(axes) < 4:
                ax = axes[0]

                if hasattr(subsample, "__iter__"):
                    raise ValueError(
                        "You have selected an iterable subsampling factor for a\
                                      single axis. Behaviour undefined in this situation."
                    )

                slicer = self.subsample_vector_field(self.__axis, self.__cut, subsample)
                vector_nda_slice = vector_nda.__getitem__(slicer)

                vector_ax = vector_nda_slice[:, :, 2].T
                vector_cor = vector_nda_slice[:, :, 1].T
                vector_sag = vector_nda_slice[:, :, 0].T

                (
                    vector_plot_x,
                    vector_plot_y,
                    vector_plot_z,
                ) = self.reorientate_vector_field(
                    self.__axis,
                    vector_ax,
                    vector_cor,
                    vector_sag,
                    invert_field=invert_field,
                )

                plot_x_loc, plot_y_loc = self.vector_image_grid(
                    self.__axis, vector_nda, subsample
                )

                if color_function == "perpendicular":
                    vector_color = vector_plot_z
                elif color_function == "magnitude":
                    vector_color = np.sqrt(
                        vector_plot_x ** 2 + vector_plot_y ** 2 + vector_plot_z ** 2
                    )

                ax.quiver(
                    plot_x_loc,
                    plot_y_loc,
                    vector_plot_x,
                    vector_plot_y,
                    vector_color,
                    cmap=colormap,
                    units="xy",
                    scale=1 / arrow_scale,
                    width=arrow_width,
                    minlength=0,
                    linewidth=1,
                )

                # if self.__show_colorbar:
                #     divider = make_axes_locatable(ax)
                #     cax = divider.append_axes("right", size="5%", pad=0.05)
                #     cbar = self.__figure.colorbar(sp, cax=cax, orientation="vertical")
                #     cbar.set_label("Probability", fontsize=16)

                #     fX, fY = self.__figure.get_size_inches()
                #     self.__figure.set_size_inches(fX * 1.15, fY)
                #     self.__figure.subplots_adjust(left=0, right=0.88, bottom=0, top=1)

            elif len(axes) == 4:
                ax_ax, _, ax_cor, ax_sag = axes

                for plot_axes, im_axis, im_cut in zip(
                    (ax_ax, ax_cor, ax_sag), ("z", "y", "x"), self.__cut
                ):

                    slicer = self.subsample_vector_field(im_axis, im_cut, subsample)
                    vector_nda_slice = vector_nda.__getitem__(slicer)

                    vector_ax = vector_nda_slice[:, :, 2].T
                    vector_cor = vector_nda_slice[:, :, 1].T
                    vector_sag = vector_nda_slice[:, :, 0].T

                    (
                        vector_plot_x,
                        vector_plot_y,
                        vector_plot_z,
                    ) = self.reorientate_vector_field(
                        im_axis, vector_ax, vector_cor, vector_sag
                    )

                    plot_x_loc, plot_y_loc = self.vector_image_grid(
                        im_axis, vector_nda, subsample
                    )

                    if color_function == "perpendicular":
                        vector_color = vector_plot_z
                    elif color_function == "magnitude":
                        vector_color = np.sqrt(
                            vector_plot_x ** 2 + vector_plot_y ** 2 + vector_plot_z ** 2
                        )

                    sp = plot_axes.quiver(
                        plot_x_loc,
                        plot_y_loc,
                        vector_plot_x,
                        vector_plot_y,
                        vector_color,
                        cmap=colormap,
                        units="xy",
                        scale=1 / arrow_scale,
                        width=arrow_width,
                        minlength=0,
                        linewidth=1,
                    )

    def overlay_bounding_boxes(self, color="r"):
        """Overlay bounding boxes onto existing figure

        Args:
            color (str|list|tuple, optional): Color of bounding box. Defaults to "r".
        """

        for box in self.__bounding_boxes:
            sag0, cor0, ax0, sagD, corD, axD = box.bounding_box

            if box.color:
                color = box.color

            # Test types of axes
            axes = self.__figure.axes[:4]
            if len(axes) < 4:
                ax = axes[0]

                if self.__axis == "z" or self.__axis == "ax":
                    ax.plot(
                        [sag0, sag0, sag0 + sagD, sag0 + sagD, sag0],
                        [cor0, cor0 + corD, cor0 + corD, cor0, cor0],
                        lw=2,
                        c=color,
                    )
                if self.__axis == "y" or self.__axis == "cor":
                    ax.plot(
                        [sag0, sag0 + sagD, sag0 + sagD, sag0, sag0],
                        [ax0, ax0, ax0 + axD, ax0 + axD, ax0],
                        lw=2,
                        c=color,
                    )
                if self.__axis == "x" or self.__axis == "sag":
                    ax.plot(
                        [cor0, cor0 + corD, cor0 + corD, cor0, cor0],
                        [ax0, ax0, ax0 + axD, ax0 + axD, ax0],
                        lw=2,
                        c=color,
                    )

            elif len(axes) == 4:
                ax_ax, _, ax_cor, ax_sag = axes

                ax_ax.plot(
                    [sag0, sag0, sag0 + sagD, sag0 + sagD, sag0],
                    [cor0, cor0 + corD, cor0 + corD, cor0, cor0],
                    lw=2,
                    c=color,
                )
                ax_cor.plot(
                    [sag0, sag0 + sagD, sag0 + sagD, sag0, sag0],
                    [ax0, ax0, ax0 + axD, ax0 + axD, ax0],
                    lw=2,
                    c=color,
                )
                ax_sag.plot(
                    [cor0, cor0 + corD, cor0 + corD, cor0, cor0],
                    [ax0, ax0, ax0 + axD, ax0 + axD, ax0],
                    lw=2,
                    c=color,
                )
