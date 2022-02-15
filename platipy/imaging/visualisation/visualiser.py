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

import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable  # , AxesGrid, ImageGrid

import numpy as np
import SimpleITK as sitk

from loguru import logger

from platipy.imaging.utils.crop import label_to_roi

from platipy.imaging.visualisation.utils import (
    VisualiseBoundingBox,
    VisualiseContour,
    VisualiseScalarOverlay,
    VisualiseVectorOverlay,
    VisualiseComparisonOverlay,
    return_slice,
    subsample_vector_field,
    vector_image_grid,
    reorientate_vector_field,
    generate_comparison_colormix,
    project_onto_arbitrary_plane,
)

"""
This Python script comprises two contributions to the code base:
1) A bunch of helpful visualisation "helper" functions

2) A visualisation class used to generate figures of images, contours, vector fields, and more!
"""


class ImageVisualiser:
    """Class to assist with visualising images and overlaying contours, scalars and bounding
    boxes."""

    def __init__(
        self,
        image,
        cut=None,
        axis="ortho",
        window=None,
        figure_size_in=10,
        limits=None,
        colormap=plt.cm.get_cmap("Greys_r"),
        origin="normal",
        projection=False,
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
        self.__projection = projection
        self.__image_view = None
        self.__scalar_view = None
        self.__contour_colormap = None

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

    def set_limits_from_label(self, label, expansion=[0, 0, 0]):
        """Sets the limits of the axes to the bounds of the given label.

        Args:
            label (sitk.Image): The label around which to set the limits
            expansion (list, optional): Expansion (in mm) around the label. Defaults to [0, 0, 0].
        """

        (sag_size, cor_size, ax_size), (sag_0, cor_0, ax_0) = label_to_roi(
            label, expansion_mm=expansion
        )

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
        self,
        contour,
        name=None,
        color=None,
        colormap=plt.cm.get_cmap("rainbow"),
        linewidth=2,
        linestyle="solid",
        show_legend=True,
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

        self.__show_legend = show_legend

        if isinstance(contour, dict):

            if not all(map(lambda i: isinstance(i, sitk.Image), contour.values())):
                raise ValueError("When passing dict, all values must be of type SimpleITK.Image")

            for contour_name in contour:

                if isinstance(color, dict):
                    try:
                        contour_color = color[contour_name]
                    except AttributeError:
                        contour_color = None
                else:
                    contour_color = color

                visualise_contour = VisualiseContour(
                    contour[contour_name],
                    contour_name,
                    color=contour_color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                )
                self.__contours.append(visualise_contour)

        elif isinstance(contour, sitk.Image):

            # Use a default name if not specified
            if name is None:
                name = "contour"
                self.__show_legend = False

            visualise_contour = VisualiseContour(
                contour, name, color=color, linewidth=linewidth, linestyle=linestyle
            )
            self.__contours.append(visualise_contour)
        else:

            raise ValueError(
                "Contours should be represented as a dict with contour name as key "
                "and sitk.Image as value, or as an sitk.Image and passing the contour_name"
            )

        self.__contour_colormap = colormap

    def add_scalar_overlay(
        self,
        scalar_image,
        name=None,
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
                raise ValueError("When passing dict, all values must be of type SimpleITK.Image")

            for name in scalar_image:
                visualise_scalar = VisualiseScalarOverlay(
                    scalar_image[name],
                    name,
                    colormap=colormap,
                    alpha=alpha,
                    min_value=min_value,
                    max_value=max_value,
                    discrete_levels=discrete_levels,
                    mid_ticks=mid_ticks,
                    show_colorbar=show_colorbar,
                    norm=norm,
                    projection=projection,
                )
                self.__scalar_overlays.append(visualise_scalar)

        elif isinstance(scalar_image, sitk.Image):

            # Use a default name if not specified
            if not name:
                name = "input"
                self.__show_legend = False

            visualise_scalar = VisualiseScalarOverlay(
                scalar_image,
                name,
                colormap=colormap,
                alpha=alpha,
                min_value=min_value,
                max_value=max_value,
                discrete_levels=discrete_levels,
                mid_ticks=mid_ticks,
                show_colorbar=show_colorbar,
                norm=norm,
                projection=projection,
            )
            self.__scalar_overlays.append(visualise_scalar)
        else:

            raise ValueError(
                "Contours should be represented as a dict with contour name as key "
                "and sitk.Image as value, or as an sitk.Image and passing the contour_name"
            )

    def add_vector_overlay(
        self,
        vector_image,
        min_value=False,
        max_value=False,
        colormap=plt.cm.get_cmap("Spectral"),
        alpha=0.75,
        arrow_scale=0.25,
        arrow_width=1,
        subsample=4,
        color_function="perpendicular",
        show_colorbar=True,
        name=None,
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
                                            'perpendicular' - vectors colored by perpendicular
                                            value
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
                min_value=min_value,
                max_value=max_value,
                colormap=colormap,
                alpha=alpha,
                arrow_scale=arrow_scale,
                arrow_width=arrow_width,
                subsample=subsample,
                color_function=color_function,
                show_colorbar=show_colorbar,
                name=name,
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

            visualise_comparison = VisualiseComparisonOverlay(
                image, name, color_rotation=color_rotation
            )
            self.__comparison_overlays.append(visualise_comparison)
        else:

            raise ValueError("Image should be sitk.Image.")

    def add_bounding_box(self, bounding_box, name=None, color="r", linewidth=2):
        """Add a bounding box to draw.

        Args:
            bounding_box (dict|list|tuple|sitk.Image): Dictionary of bounding boxes where list key
                                    is the name of the bounding box. If list or tuple then pass
                                    bounding box in format (x, y, z, w, h, d). If sitk.Image then
                                    mask is assumed and bounding box around this mask is used.
            name (list, optional): Name of this bounding box if bounding_box is list, tuple or
                                   sitk.Image. Defaults to None.
            color (str|tuple|list, optional): The color to use when drawing the bounding box(es).
                                              Defaults to None.
            linewidth (int, optional): The line width of the bounding box(es). Defaults to 2.

        Raises:
            ValueError: Raised if input is not a list or tuple of length 6 or and sitk.Image.
        """

        self.__show_legend = True

        if isinstance(bounding_box, dict):

            if not all(
                map(
                    lambda i: isinstance(i, (list, tuple, sitk.Image)) and len(i) == 6,
                    bounding_box.values(),
                )
            ):
                raise ValueError("All values must be of type list or tuple with length 6")

            for name in bounding_box:
                visualise_bounding_box = VisualiseBoundingBox(
                    bounding_box[name], name=name, color=color, linewidth=linewidth
                )
                self.__bounding_boxes.append(visualise_bounding_box)

        elif isinstance(bounding_box, (list, tuple, sitk.Image)):

            # Use a default name if not specified
            if name is None:
                name = "Bounding box"

            visualise_bounding_box = VisualiseBoundingBox(
                bounding_box, name=name, color=color, linewidth=linewidth
            )
            self.__bounding_boxes.append(visualise_bounding_box)

        else:
            raise ValueError(
                "Bounding boxes should be represented as a dict with bounding box name as key "
                "and list or tuple as value"
            )

    def show(self, interact=False):
        """Render the image with all overlays"""
        if len(self.__comparison_overlays) == 0:
            self._display_slice()
        else:
            self._overlay_comparison()

        self._overlay_scalar_field()
        self._overlay_vector_field()
        self._overlay_contours()
        self._overlay_bounding_boxes()

        self._adjust_view()

        if interact:
            logger.warning("Interactive mode not yet implemented")
            # self.interact_adjust_slice()

        self.__figure.canvas.draw()
        self._add_legend()
        self.__figure.set_facecolor("white")

        return self.__figure

    def _display_slice(self):
        """Display the configured image slice"""

        image = self.__image
        nda = sitk.GetArrayFromImage(image)

        (ax_size, cor_size, sag_size) = nda.shape[:3]

        window = self.__window
        if window is None:
            # We will choose it ourselves!

            lower = nda.min()

            # Check if we *probably* have a CT
            if lower < -900:
                # Just set a decent CT window
                # Somewhere around soft tissue
                window = (-250, 600)

            # Otherwise just pick a reasonable upper limit
            else:
                upper = np.percentile(nda, 99)
                window = (lower, upper - lower)
        try:
            logger.info(
                f"Found a (z,y,x,{nda.shape[3]}) dimensional array - assuming this is an RGB"
                "image."
            )
            nda /= nda.max()
        except ValueError:
            logger.warning("Problem converting RGB image to np.ndarray.")
        except IndexError:
            pass

        sp_plane, _, sp_slice = image.GetSpacing()
        asp = (1.0 * sp_slice) / sp_plane

        if self.__projection is True:
            projection = "max"
        else:
            projection = self.__projection

        if self.__axis == "ortho":
            figure_size = (
                self.__figure_size,
                self.__figure_size * (asp * ax_size + cor_size) / (1.0 * sag_size + cor_size),
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

            if not self.__projection:
                s_ax = return_slice("z", self.__cut[0])
                s_cor = return_slice("y", self.__cut[1])
                s_sag = return_slice("x", self.__cut[2])

                ax_img = nda.__getitem__(s_ax)
                cor_img = nda.__getitem__(s_cor)
                sag_img = nda.__getitem__(s_sag)

            else:
                ax_img_proj = project_onto_arbitrary_plane(
                    image,
                    projection_axis=2,
                    projection_name=projection,
                    default_value=int(nda.min()),
                )
                ax_img = sitk.GetArrayFromImage(ax_img_proj)
                # ax_img = (ax_img - ax_img.min()) / (ax_img.max() - ax_img.min())

                cor_img_proj = project_onto_arbitrary_plane(
                    image,
                    projection_axis=1,
                    projection_name=projection,
                    default_value=int(nda.min()),
                )
                cor_img = sitk.GetArrayFromImage(cor_img_proj)
                # cor_img = (cor_img - cor_img.min()) / (cor_img.max() - cor_img.min())

                sag_img_proj = project_onto_arbitrary_plane(
                    image,
                    projection_axis=0,
                    projection_name=projection,
                    default_value=int(nda.min()),
                )
                sag_img = sitk.GetArrayFromImage(sag_img_proj)
                # sag_img = (sag_img - sag_img.min()) / (sag_img.max() - sag_img.min())

            ax_view = ax_ax.imshow(
                ax_img,
                aspect=1.0,
                interpolation="none",
                origin={"normal": "upper", "reversed": "lower"}[self.__origin],
                cmap=self.__colormap,
                clim=(window[0], window[0] + window[1]),
            )
            cor_view = ax_cor.imshow(
                cor_img,
                origin="lower",
                aspect=asp,
                interpolation="none",
                cmap=self.__colormap,
                clim=(window[0], window[0] + window[1]),
            )
            sag_view = ax_sag.imshow(
                sag_img,
                origin="lower",
                aspect=asp,
                interpolation="none",
                cmap=self.__colormap,
                clim=(window[0], window[0] + window[1]),
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
                    "You have selected a single axis and multiple slice locations, attempting to "
                    "match."
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

            if not self.__projection:
                s = return_slice(self.__axis, self.__cut)
                disp_img = nda.__getitem__(s)
            else:
                disp_img_proj = project_onto_arbitrary_plane(
                    image,
                    projection_axis={"x": 0, "y": 1, "z": 2}[self.__axis],
                    projection_name=projection,
                    default_value=int(nda.min()),
                )
                disp_img = sitk.GetArrayFromImage(disp_img_proj)
                # disp_img = (disp_img - disp_img.min()) / (disp_img.max() - disp_img.min())

            s = return_slice(self.__axis, self.__cut)
            ax_indiv = ax.imshow(
                disp_img,
                aspect=asp,
                interpolation="none",
                origin=org,
                cmap=self.__colormap,
                clim=(window[0], window[0] + window[1]),
            )
            ax.axis("off")

            self.__figure.subplots_adjust(left=0, right=1, bottom=0, top=1)

            self.__image_view = {axis_view_name_consistent: ax_indiv}

    def _overlay_comparison(self):
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
        if window is None:
            # We will choose it ourselves!

            lower = nda_original.min()

            # Check if we *probably* have a CT
            if lower < -900:
                # Just set a decent CT window
                # Somewhere around soft tissue
                window = (-250, 600)

            # Otherwise just pick a reasonable upper limit
            else:
                upper = np.percentile(nda_original, 99)
                window = (lower, upper - lower)

        if self.__axis == "ortho":
            figure_size = (
                self.__figure_size,
                self.__figure_size * (asp * ax_size + cor_size) / (1.0 * sag_size + cor_size),
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

            s_ax = return_slice("z", self.__cut[0])
            s_cor = return_slice("y", self.__cut[1])
            s_sag = return_slice("x", self.__cut[2])

            nda_colormix = generate_comparison_colormix(
                [nda_original, nda_new],
                arr_slice=s_ax,
                window=window,
                color_rotation=color_rotation,
            )

            ax_ax.imshow(
                nda_colormix,
                aspect=1.0,
                origin={"normal": "upper", "reversed": "lower"}[self.__origin],
                interpolation="none",
            )

            nda_colormix = generate_comparison_colormix(
                [nda_original, nda_new],
                arr_slice=s_cor,
                window=window,
                color_rotation=color_rotation,
            )

            ax_cor.imshow(
                nda_colormix,
                origin="lower",
                aspect=asp,
                interpolation="none",
            )

            nda_colormix = generate_comparison_colormix(
                [nda_original, nda_new],
                arr_slice=s_sag,
                window=window,
                color_rotation=color_rotation,
            )

            ax_sag.imshow(
                nda_colormix,
                origin="lower",
                aspect=asp,
                interpolation="none",
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
                    "You have selected a single axis and multiple slice locations, attempting to "
                    "match."
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

            s = return_slice(self.__axis, self.__cut)

            nda_colormix = generate_comparison_colormix(
                [nda_original, nda_new], arr_slice=s, window=window, color_rotation=color_rotation
            )

            ax.imshow(
                nda_colormix,
                aspect=asp,
                interpolation="none",
                origin=org,
            )
            ax.axis("off")

            self.__figure.subplots_adjust(left=0, right=1, bottom=0, top=1)

    def _adjust_view(self):
        """adjust_view is a helper function for modifying axis limits.
        Specify *limits* when initialising the ImageVisulaiser to use.
        Alternatively, use set_limits_from_label to specify automatically.
        """

        limits = self.__limits
        origin = self.__origin

        if limits is not None:
            if self.__axis == "ortho":
                ax_ax, ax_blank, ax_cor, ax_sag = self.__figure.axes[:4]
                cax_list = self.__figure.axes[4:]

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

                if origin == "reversed":
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

                ax_blank.set_position(gs[1].get_position(self.__figure))
                ax_blank.set_subplotspec(gs[1])

                ax_cor.set_position(gs[2].get_position(self.__figure))
                ax_cor.set_subplotspec(gs[2])

                ax_sag.set_position(gs[3].get_position(self.__figure))
                ax_sag.set_subplotspec(gs[3])

                fig_size_x, fig_size_y = self.__figure.get_size_inches()
                fig_size_y = fig_size_y * ratio_y / ratio_x

                ax_ax_bbox = gs[0].get_position(self.__figure)

                for cax_index, cax in enumerate(cax_list):

                    cbar_width = ax_ax_bbox.width * 0.05

                    cax.set_position(
                        (
                            ax_ax_bbox.x1 + 0.02 + (cbar_width + 0.1) * cax_index,
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

    def _overlay_contours(self):
        """Overlay the contours on to the current figure image"""

        if len(self.__contours) == 0:
            return

        plot_dict = {}
        color_dict = {}
        lw_dict = {}
        ls_dict = {}

        color_gen_index = 0

        for contour in self.__contours:
            contour_image_resampled = sitk.Resample(contour.image, self.__image)
            plot_dict[contour.name] = contour_image_resampled

            if contour.color is not None:
                color_dict[contour.name] = contour.color
            else:
                color_map = self.__contour_colormap(np.linspace(0, 1, len(self.__contours)))

                color_dict[contour.name] = color_map[color_gen_index % 255]
                color_gen_index += 1

            lw_dict[contour.name] = contour.linewidth
            ls_dict[contour.name] = contour.linestyle

        # Test types of axes
        axes = self.__figure.axes[:4]

        if self.__axis in ["x", "y", "z"]:
            ax = axes[0]
            s = return_slice(self.__axis, self.__cut)

            for c_name in plot_dict:
                if not self.__projection:
                    contour_disp = sitk.GetArrayFromImage(plot_dict[c_name]).__getitem__(s)

                    # Force a single pixel to 1 to display all contours
                    # even if they aren't defined on a particular slice
                    if contour_disp.sum() == 0:
                        contour_disp[0, 0] = 1

                else:
                    contour_disp_proj = project_onto_arbitrary_plane(
                        plot_dict[c_name],
                        projection_axis={"x": 0, "y": 1, "z": 2}[self.__axis],
                        projection_name="max",
                        default_value=0,
                    )
                    contour_disp = sitk.GetArrayFromImage(contour_disp_proj)

                try:
                    temp = ax.contour(
                        contour_disp,
                        colors=[color_dict[c_name]],
                        levels=[0.5],
                        # alpha=0.8,
                        linewidths=lw_dict[c_name],
                        linestyles=ls_dict[c_name],
                        label=c_name,
                        origin="lower",
                    )
                    temp.collections[0].set_label(c_name)
                except AttributeError:
                    pass

        elif self.__axis == "ortho":
            ax_ax, _, ax_cor, ax_sag = axes

            ax = ax_ax

            s_ax = return_slice("z", self.__cut[0])
            s_cor = return_slice("y", self.__cut[1])
            s_sag = return_slice("x", self.__cut[2])

            for _, c_name in enumerate(plot_dict.keys()):

                if not self.__projection:

                    contour_ax = sitk.GetArrayFromImage(plot_dict[c_name]).__getitem__(s_ax)
                    contour_cor = sitk.GetArrayFromImage(plot_dict[c_name]).__getitem__(s_cor)
                    contour_sag = sitk.GetArrayFromImage(plot_dict[c_name]).__getitem__(s_sag)

                    # Force a single pixel to 1 to display all contours
                    # even if they aren't defined on a particular slice
                    if contour_ax.sum() == 0:
                        contour_ax[0, 0] = 1
                    if contour_cor.sum() == 0:
                        contour_cor[0, 0] = 1
                    if contour_sag.sum() == 0:
                        contour_sag[0, 0] = 1

                else:
                    contour_ax_proj = project_onto_arbitrary_plane(
                        plot_dict[c_name],
                        projection_axis=2,
                        projection_name="max",
                        default_value=0,
                    )
                    contour_ax = sitk.GetArrayFromImage(contour_ax_proj)

                    contour_cor_proj = project_onto_arbitrary_plane(
                        plot_dict[c_name],
                        projection_axis=1,
                        projection_name="max",
                        default_value=0,
                    )
                    contour_cor = sitk.GetArrayFromImage(contour_cor_proj)

                    contour_sag_proj = project_onto_arbitrary_plane(
                        plot_dict[c_name],
                        projection_axis=0,
                        projection_name="max",
                        default_value=0,
                    )
                    contour_sag = sitk.GetArrayFromImage(contour_sag_proj)

                temp = ax_ax.contour(
                    contour_ax,
                    levels=[0.5],
                    linewidths=lw_dict[c_name],
                    linestyles=ls_dict[c_name],
                    colors=[color_dict[c_name]],
                    origin="lower",
                )
                temp.collections[0].set_label(c_name)

                ax_cor.contour(
                    contour_cor,
                    levels=[0.5],
                    linewidths=lw_dict[c_name],
                    linestyles=ls_dict[c_name],
                    colors=[color_dict[c_name]],
                    origin="lower",
                )
                ax_sag.contour(
                    contour_sag,
                    levels=[0.5],
                    linewidths=lw_dict[c_name],
                    linestyles=ls_dict[c_name],
                    colors=[color_dict[c_name]],
                    origin="lower",
                )

        else:
            raise ValueError('Axis is must be one of "x","y","z","ortho".')

    def _overlay_scalar_field(self):
        """Overlay the scalar image onto the existing figure"""

        for scalar_index, scalar in enumerate(self.__scalar_overlays):

            scalar_image = scalar.image
            nda = sitk.GetArrayFromImage(scalar_image)

            alpha = scalar.alpha

            if scalar.max_value is not False:
                s_max = scalar.max_value
            else:
                s_max = nda.max()

            if scalar.min_value is not False:
                s_min = scalar.min_value
            else:
                s_min = nda.min()

            if scalar.discrete_levels:
                colormap_name = scalar.colormap.name
                colormap = plt.cm.get_cmap(colormap_name, scalar.discrete_levels)

            else:
                colormap = scalar.colormap

            if scalar.norm:
                norm = scalar.norm
            else:
                norm = None

            sp_plane, _, sp_slice = scalar_image.GetSpacing()

            asp = (1.0 * sp_slice) / sp_plane

            # projection organisation
            if scalar.projection:
                projection = scalar.projection
            else:
                projection = self.__projection

            if self.__axis == "ortho":

                ax_ax, _, ax_cor, ax_sag = self.__figure.axes
                ax = ax_ax

                if not projection:
                    s_ax = return_slice("z", self.__cut[0])
                    s_cor = return_slice("y", self.__cut[1])
                    s_sag = return_slice("x", self.__cut[2])

                    ax_img = nda.__getitem__(s_ax)
                    cor_img = nda.__getitem__(s_cor)
                    sag_img = nda.__getitem__(s_sag)

                else:
                    ax_img_proj = project_onto_arbitrary_plane(
                        scalar_image,
                        projection_axis=2,
                        projection_name=projection,
                        default_value=int(nda.min()),
                    )
                    ax_img = sitk.GetArrayFromImage(ax_img_proj)

                    cor_img_proj = project_onto_arbitrary_plane(
                        scalar_image,
                        projection_axis=1,
                        projection_name=projection,
                        default_value=int(nda.min()),
                    )
                    cor_img = sitk.GetArrayFromImage(cor_img_proj)

                    sag_img_proj = project_onto_arbitrary_plane(
                        scalar_image,
                        projection_axis=0,
                        projection_name=projection,
                        default_value=int(nda.min()),
                    )
                    sag_img = sitk.GetArrayFromImage(sag_img_proj)

                # mask images to enforce transparency
                ax_img = np.ma.masked_less_equal(ax_img, s_min)
                cor_img = np.ma.masked_less_equal(cor_img, s_min)
                sag_img = np.ma.masked_less_equal(sag_img, s_min)

                ax_view = ax_ax.imshow(
                    ax_img,
                    interpolation="none",
                    cmap=colormap,
                    clim=(s_min, s_max),
                    aspect=1,
                    origin={"normal": "upper", "reversed": "lower"}[self.__origin],
                    vmin=s_min,
                    vmax=s_max,
                    alpha=alpha,
                    norm=norm,
                )

                cor_view = ax_cor.imshow(
                    cor_img,
                    interpolation="none",
                    cmap=colormap,
                    clim=(s_min, s_max),
                    origin="lower",
                    aspect=asp,
                    vmin=s_min,
                    vmax=s_max,
                    alpha=alpha,
                    norm=norm,
                )

                sag_view = ax_sag.imshow(
                    sag_img,
                    interpolation="none",
                    cmap=colormap,
                    clim=(s_min, s_max),
                    origin="lower",
                    aspect=asp,
                    vmin=s_min,
                    vmax=s_max,
                    alpha=alpha,
                    norm=norm,
                )

                # this is for (work-in-progress) dynamic visualisation
                self.__scalar_view = {
                    "ax_view": ax_view,
                    "cor_view": cor_view,
                    "sag_view": sag_view,
                }

            else:

                ax = self.__figure.axes[0]

                if not projection:
                    s = return_slice(self.__axis, self.__cut)
                    disp_img = nda.__getitem__(s)
                else:
                    disp_img_proj = project_onto_arbitrary_plane(
                        scalar_image,
                        projection_axis={"x": 0, "y": 1, "z": 2}[self.__axis],
                        projection_name=projection,
                        default_value=int(nda.min()),
                    )
                    disp_img = sitk.GetArrayFromImage(disp_img_proj)
                    # disp_img = (disp_img - disp_img.min()) / (disp_img.max() - disp_img.min())

                disp_img = np.ma.masked_less_equal(disp_img, s_min)

                asp = {"x": asp, "y": asp, "z": 1}[self.__axis]

                s = return_slice(self.__axis, self.__cut)
                ax_view = ax.imshow(
                    disp_img,
                    interpolation="none",
                    cmap=colormap,
                    clim=(s_min, s_max),
                    origin="lower",
                    aspect=asp,
                    vmin=s_min,
                    vmax=s_max,
                    alpha=alpha,
                    norm=norm,
                )

            if scalar.show_colorbar:

                # divider = make_axes_locatable(ax_view)
                # cax = divider.append_axes("right", size="5%", pad=0.05)

                ax_box = ax.get_position(original=False)
                cbar_width = ax_box.width * 0.05  # 5% of axis width
                cbar_color = "black"

                if self.__axis == "ortho":
                    cax = self.__figure.add_axes(
                        (
                            ax_box.x1 + 0.02 + (cbar_width + 0.1) * scalar_index,
                            ax_box.y0 * 1.025,
                            cbar_width,
                            ax_box.height - ax_box.y0 * 0.05,
                        )
                    )

                else:
                    cax = self.__figure.add_axes(
                        (
                            ax_box.x1 - 0.02 - (cbar_width + 0.1) * (scalar_index + 1),
                            0.025,
                            cbar_width,
                            ax_box.height - ax_box.y1 * 0.05,
                        )
                    )

                    # check background values
                    if np.linalg.norm(colormap(0)[:3]) < 0.1:
                        # background is dark
                        cbar_color = "white"

                cbar = self.__figure.colorbar(ax_view, cax=cax, orientation="vertical")

                # set color
                cbar.outline.set_edgecolor(color=cbar_color)
                cbar.ax.tick_params(color=cbar_color)
                cax.tick_params(axis="x", colors=cbar_color)
                cax.tick_params(axis="y", colors=cbar_color)

                cbar.set_label(scalar.name, color=cbar_color)
                cbar.solids.set_alpha(1)

                if scalar.discrete_levels:

                    if scalar.mid_ticks:

                        delta_tick = (s_max - s_min) / scalar.discrete_levels
                        cbar.set_ticks(
                            np.linspace(
                                s_min + delta_tick / 2,
                                s_max - delta_tick / 2,
                                scalar.discrete_levels,
                            )
                        )
                        cbar.set_ticklabels(np.linspace(s_min, s_max, scalar.discrete_levels))

                    else:
                        cbar.set_ticks(
                            np.linspace(
                                s_min,
                                s_max,
                                scalar.discrete_levels + 1,
                            )
                        )

    def _overlay_vector_field(self):
        """Overlay vector field onto existing figure"""

        if self.__projection and len(self.__vector_overlays) > 0:
            raise Warning("Vector overlay is not implemented in projection mode.")

        for _, vector in enumerate(self.__vector_overlays):

            image = vector.image
            colormap = vector.colormap
            alpha = vector.alpha
            arrow_scale = vector.arrow_scale
            arrow_width = vector.arrow_width
            subsample = vector.subsample
            color_function = vector.color_function
            invert_field = vector.invert_field
            alpha = vector.alpha
            min_value = vector.min_value
            max_value = vector.max_value

            inverse_vector_image = image  # sitk.InvertDisplacementField(image)
            vector_nda = sitk.GetArrayFromImage(inverse_vector_image)

            # Test types of axes
            axes = self.__figure.axes
            if len(axes[:4]) < 4:
                ax = axes[0]

                if hasattr(subsample, "__iter__"):
                    raise ValueError(
                        "You have selected an iterable subsampling factor for a\
                                      single axis. Behaviour undefined in this situation."
                    )

                slicer = subsample_vector_field(self.__axis, self.__cut, subsample)
                vector_nda_slice = vector_nda.__getitem__(slicer)

                vector_ax = vector_nda_slice[:, :, 2].T
                vector_cor = vector_nda_slice[:, :, 1].T
                vector_sag = vector_nda_slice[:, :, 0].T

                (vector_plot_x, vector_plot_y, vector_plot_z,) = reorientate_vector_field(
                    self.__axis,
                    vector_ax,
                    vector_cor,
                    vector_sag,
                    invert_field=invert_field,
                )

                plot_x_loc, plot_y_loc = vector_image_grid(self.__axis, vector_nda, subsample)

                if color_function == "perpendicular":
                    vector_color = vector_plot_z
                elif color_function == "magnitude":
                    vector_color = np.sqrt(
                        vector_plot_x ** 2 + vector_plot_y ** 2 + vector_plot_z ** 2
                    )

                if ~max_value:
                    max_value = vector_color.max()
                if ~min_value:
                    min_value = vector_color.min()

                sp_vector = ax.quiver(
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
                    alpha=alpha,
                    clim=[min_value, max_value],
                )

                if vector.show_colorbar:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = self.__figure.colorbar(sp_vector, cax=cax, orientation="vertical")
                    cbar.set_label(vector.name)
                    cbar.solids.set_alpha(1)

                    f_x, f_y = self.__figure.get_size_inches()
                    self.__figure.set_size_inches(f_x * 1.15, f_y)
                    self.__figure.subplots_adjust(left=0, right=0.88, bottom=0, top=1)

            elif len(axes) >= 4:
                ax_ax, _, ax_cor, ax_sag = axes[:4]

                for plot_axes, im_axis, im_cut in zip(
                    (ax_ax, ax_cor, ax_sag), ("z", "y", "x"), self.__cut
                ):

                    slicer = subsample_vector_field(im_axis, im_cut, subsample)
                    vector_nda_slice = vector_nda.__getitem__(slicer)

                    vector_ax = vector_nda_slice[:, :, 2].T
                    vector_cor = vector_nda_slice[:, :, 1].T
                    vector_sag = vector_nda_slice[:, :, 0].T

                    (
                        vector_plot_x,
                        vector_plot_y,
                        vector_plot_z,
                    ) = reorientate_vector_field(im_axis, vector_ax, vector_cor, vector_sag)

                    plot_x_loc, plot_y_loc = vector_image_grid(im_axis, vector_nda, subsample)

                    if color_function == "perpendicular":
                        vector_color = vector_plot_z
                    elif color_function == "magnitude":
                        vector_color = np.sqrt(
                            vector_plot_x ** 2 + vector_plot_y ** 2 + vector_plot_z ** 2
                        )

                    if max_value == False:
                        max_value = vector_color.max()
                    if min_value == False:
                        min_value = vector_color.min()

                    sp_vector = plot_axes.quiver(
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
                        clim=[min_value, max_value],
                    )

                if vector.show_colorbar:

                    ax_box = ax_ax.get_position(original=False)
                    cbar_width = ax_box.width * 0.05  # 5% of axis width

                    if len(self.__figure.axes) >= 5:
                        # There is a colorbar, add a new one

                        cbar_axes = self.__figure.axes[4:]

                        cbar_axes_label_pos_list = [
                            self.__figure.transFigure.inverted().transform(
                                i.yaxis.get_label().get_position()
                            )[0]
                            for i in cbar_axes
                        ]

                        max_xpos = max(cbar_axes_label_pos_list)

                        x_pos_legend = max_xpos + 0.025

                    else:
                        x_pos_legend = ax_box.x1 + 0.025

                    cax = self.__figure.add_axes(
                        (
                            x_pos_legend,
                            ax_box.y0 * 1.025,
                            cbar_width,
                            ax_box.height - ax_box.y0 * 0.05,
                        )
                    )

                    cbar = self.__figure.colorbar(sp_vector, cax=cax, orientation="vertical")
                    cbar.set_label(vector.name)

    def draw_bounding_box_on_axes(self, ax, view, box, add_label=True):

        sag_0, cor_0, ax_0, sag_d, cor_d, ax_d = box.bounding_box

        from_points = None
        to_points = None

        if view == "z" or view == "ax":
            from_points = [sag_0, sag_0, sag_0 + sag_d, sag_0 + sag_d, sag_0]
            to_points = [cor_0, cor_0 + cor_d, cor_0 + cor_d, cor_0, cor_0]

        if view == "y" or view == "cor":
            from_points = [sag_0, sag_0 + sag_d, sag_0 + sag_d, sag_0, sag_0]
            to_points = [ax_0, ax_0, ax_0 + ax_d, ax_0 + ax_d, ax_0]

        if view == "x" or view == "sag":
            from_points = [cor_0, cor_0 + cor_d, cor_0 + cor_d, cor_0, cor_0]
            to_points = [ax_0, ax_0, ax_0 + ax_d, ax_0 + ax_d, ax_0]

        if from_points is None:
            logger.error("Invalid View")
            return

        ax.plot(
            from_points,
            to_points,
            lw=box.linewidth,
            c=box.color,
            label=box.name if add_label else None,
        )

    def _overlay_bounding_boxes(self):
        """Overlay bounding boxes onto existing figure

        Args:
            color (str|list|tuple, optional): Color of bounding box. Defaults to "r".
        """

        for box in self.__bounding_boxes:

            # Test types of axes
            axes = self.__figure.axes[:4]
            if len(axes) < 4:
                self.draw_bounding_box_on_axes(axes[0], self.__axis, box)

            elif len(axes) == 4:
                ax_ax, _, ax_cor, ax_sag = axes

                self.draw_bounding_box_on_axes(ax_ax, "z", box)
                self.draw_bounding_box_on_axes(ax_cor, "y", box, add_label=False)
                self.draw_bounding_box_on_axes(ax_sag, "x", box, add_label=False)

    def _add_legend(self):
        """Add a legend to the visualisation"""

        if len(self.__figure.axes) >= 4:

            ax_ax = self.__figure.axes[0]
            ax_ax_position = ax_ax.get_position()
            y_pos_legend = (ax_ax_position.ymax + ax_ax_position.ymin) / 2

            if self.__show_legend:
                if len(self.__figure.axes) >= 5:
                    # There is a colorbar or more - right align instead

                    cbar_axes = self.__figure.axes[4:]

                    cbar_axes_label_pos_list = [
                        self.__figure.transFigure.inverted().transform(
                            i.yaxis.get_label().get_position()
                        )[0]
                        for i in cbar_axes
                    ]

                    max_xpos = max(cbar_axes_label_pos_list)

                    x_pos_legend = max_xpos + 0.025

                    approx_font_scaling = self.__figure_size / (
                        len(self.__contours) + len(self.__bounding_boxes)
                    )

                    plt.figlegend(
                        loc="center left",
                        bbox_to_anchor=(x_pos_legend, y_pos_legend),
                        fontsize=min([10, 16 * approx_font_scaling]),
                    )
                else:
                    # Left align to axial slice
                    ax_ax_position = ax_ax.get_position()
                    x_pos_legend = ax_ax_position.xmax + 0.05

                    approx_font_scaling = self.__figure_size / (
                        len(self.__contours) + len(self.__bounding_boxes)
                    )

                    plt.figlegend(
                        loc="center left",
                        bbox_to_anchor=(x_pos_legend, y_pos_legend),
                        fontsize=min([10, 16 * approx_font_scaling]),
                    )
        else:
            # these is probably only one axis
            if self.__show_legend:
                ax = self.__figure.axes[0]
                ax_position = ax.get_position()

                # top left corner is the default
                y_pos_legend = 0.975 * ax_position.ymax
                x_pos_legend = 0.025 * ax_position.xmax

                approx_font_scaling = self.__figure_size / (
                    len(self.__contours) + len(self.__bounding_boxes)
                )

                plt.figlegend(
                    loc="upper left",
                    bbox_to_anchor=(x_pos_legend, y_pos_legend),
                    fontsize=min([10, 16 * approx_font_scaling]),
                )
