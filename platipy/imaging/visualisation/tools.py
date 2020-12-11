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


import SimpleITK as sitk
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable  # , AxesGrid, ImageGrid

import matplotlib.gridspec as gridspec

from skimage.color import hsv2rgb

class VisualiseContour:
    """Class to represent the visualiation of a contour
    """

    def __init__(self, image, name, color=None):
        self.image = image
        self.name = name
        self.color = color

class VisualiseScalarOverlay:
    """Class to represent the visualiation of a scalar overlay
    """

    def __init__(
        self, image, name, colormap=plt.cm.get_cmap("Spectral"), alpha=0.75, min_value=0.1
    ):
        self.image = image
        self.name = name
        self.colormap = colormap
        self.alpha = alpha
        self.min_value = min_value

class VisualiseVectorOverlay:
    """Class to represent the visualiation of a vector overlay
    """

    def __init__(
        self, image, name, colormap=plt.cm.get_cmap("Spectral"), alpha=0.75, arrow_scale=0.25,
        arrow_width=1, subsample=4, color_function='perpendicular'
    ):
        self.image = image
        self.name = name
        self.colormap = colormap
        self.alpha = alpha
        self.arrow_scale = arrow_scale
        self.arrow_width = arrow_width
        self.subsample = subsample
        self.color_function = color_function

class VisualiseComparisonOverlay:
    """Class to represent the visualiation of a comparison image
    """

    def __init__(
        self, image, name, color_rotation=0.35
    ):
        self.image = image
        self.color_rotation = color_rotation

class VisualiseBoundingBox:
    """Class to represent the visualiation of a bounding box
    """

    def __init__(self, bounding_box, name, color=None):
        self.bounding_box = bounding_box
        self.name = name
        self.color = color

class ImageVisualiser:
    """Class to assist with visualising images and overlaying contours, scalars and bounding boxes.
    """

    def __init__(self, image, cut=None, axis="ortho", window=[-250, 500], figure_size_in=10, limits=None):
        self.__set_image(image)
        self.__contours = []
        self.__bounding_boxes = []
        self.__scalar_overlays = []
        self.__vector_overlays = []
        self.__comparison_overlays = []
        self.__contour_color_base = plt.cm.get_cmap("Blues")
        self.__show_legend = False
        self.__show_colorbar = False
        self.__figure = None
        self.__figure_size = figure_size_in
        self.__window = window
        self.__axis = axis
        self.__cut = cut
        self.__limits = limits

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
        """Clear all overlays
        """

        self.__contours = []
        self.__bounding_boxes = []
        self.__scalar_overlays = []
        self.__comparison_overlays = []
        self.__vector_overlays = []

    def add_contour(self, contour, name=None, color=None):
        """Add a contour to overlay

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

        self.__show_legend = True

        if isinstance(contour, dict):

            if not all(map(lambda i: isinstance(i, sitk.Image), contour.values())):
                raise ValueError("When passing dict, all values must be of type SimpleITK.Image")

            for contour_name in contour:
                visualise_contour = VisualiseContour(
                    contour[contour_name], contour_name, color=None
                )
                self.__contours.append(visualise_contour)

        elif isinstance(contour, sitk.Image):

            # Use a default name if not specified
            if not name:
                name = "input"
                self.__show_legend = False

            visualise_contour = VisualiseContour(contour, name, color=color)
            self.__contours.append(visualise_contour)
        else:

            raise ValueError(
                "Contours should be represented as a dict with contour name as key "
                "and sitk.Image as value, or as an sitk.Image and passing the contour_name"
            )

    def add_scalar_overlay(
        self,
        scalar_image,
        name=None,
        colormap=plt.cm.get_cmap("Spectral"),
        alpha=0.75,
        min_value=0.1,
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
                visualise_contour = VisualiseScalarOverlay(
                    scalar_image[name], name, colormap=colormap, alpha=alpha, min_value=min_value
                )
                self.__scalar_overlays.append(visualise_contour)

        elif isinstance(scalar_image, sitk.Image):

            # Use a default name if not specified
            if not name:
                name = "input"
                self.__show_legend = False

            visualise_contour = VisualiseScalarOverlay(
                scalar_image, name, colormap=colormap, alpha=alpha, min_value=min_value
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
        color_function='perpendicular'
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

        if isinstance(vector_image, sitk.Image) and vector_image.GetNumberOfComponentsPerPixel()>1:

            # Use a default name if not specified
            if not name:
                name = "input"
                self.__show_legend = False

            visualise_vector_field = VisualiseVectorOverlay(
                vector_image,
                name,
                colormap = colormap,
                alpha = alpha,
                arrow_scale = arrow_scale,
                arrow_width = arrow_width,
                subsample = subsample,
                color_function = color_function
            )
            self.__vector_overlays.append(visualise_vector_field)
        else:

            raise ValueError(
                "Vector field should be sitk.Image (of vector type)."
            )

    def add_comparison_overlay(
        self,
        image,
        name=None,
        color_rotation=0.35
    ):
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
                image,
                name,
                color_rotation=color_rotation
            )
            self.__comparison_overlays.append(visualise_comparison)
        else:

            raise ValueError(
                "Image should be sitk.Image."
            )

    def add_bounding_box(self, bounding_box, name=None, color=None):

        self.__show_legend = True

        if isinstance(bounding_box, dict):

            if not all(
                map(lambda i: isinstance(i, (list, tuple)) and len(i) == 6, bounding_box.values())
            ):
                raise ValueError("All values must be of type list or tuple with length 6")

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

            visualise_bounding_box = VisualiseBoundingBox(bounding_box, name, color=color)
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
            subsample_ax, subsample_cor, subsample_sag = (subsample,)*3

        if axis=='x':
            return (slice(None, None, subsample_ax), slice(None, None, subsample_cor), cut)
        if axis=='y':
            return (slice(None, None, subsample_ax), cut, slice(None, None, subsample_sag))
        if axis=='z':
            return (cut, slice(None, None, subsample_cor), slice(None, None, subsample_sag))
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
            subsample_ax, subsample_cor, subsample_sag = (subsample,)*3

        if axis=='x':
            return np.mgrid[0:vector_field_array.shape[1]:subsample_cor,0:vector_field_array.shape[0]:subsample_ax]
        if axis=='y':
            return np.mgrid[0:vector_field_array.shape[2]:subsample_sag,0:vector_field_array.shape[0]:subsample_ax]
        if axis=='z':
            return np.mgrid[0:vector_field_array.shape[2]:subsample_sag,0:vector_field_array.shape[1]:subsample_cor]
        return None

    def reorientate_vector_field(self, axis, vector_ax, vector_cor, vector_sag):
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
        if axis=='x': # sagittal projection
            return  1.0*vector_cor, 1.0*vector_ax,  1.0*vector_sag
        if axis=='y': # coronal projection
            return -1.0*vector_sag, 1.0*vector_ax,  1.0*vector_cor
        if axis=='z': # axial projection
            return -1.0*vector_sag, 1.0*vector_cor, 1.0*vector_ax
            
        return None

    def show(self):
        """Render the image with all overlays
        """
        if len(self.__comparison_overlays)==0:
            self.display_slice()
        else:
            self.overlay_comparison()

        self.overlay_scalar_field()
        self.overlay_vector_field()
        self.overlay_contours()
        self.overlay_bounding_boxes()
        self.adjust_view()

        return self.__figure

    def display_slice(self, cmap=plt.cm.get_cmap("Greys_r")):
        """Display the configured image slice

        Args:
            cmap (matplotlib.colors.Colormap, optional): The colormap to be used to display this
                                                         image. plt.cm.get_cmap("Greys_r").
        """

        image = self.__image
        nda = sitk.GetArrayFromImage(image)

        (ax_size, cor_size, sag_size) = nda.shape
        sp_plane, _, sp_slice = image.GetSpacing()
        asp = (1.0 * sp_slice) / sp_plane

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

            s_ax = self.return_slice("z", self.__cut[0])
            s_cor = self.return_slice("y", self.__cut[1])
            s_sag = self.return_slice("x", self.__cut[2])

            ax_ax.imshow(
                nda.__getitem__(s_ax),
                aspect=1.0,
                interpolation=None,
                cmap=cmap,
                clim=(self.__window[0], self.__window[0] + self.__window[1]),
            )
            ax_cor.imshow(
                nda.__getitem__(s_cor),
                origin="lower",
                aspect=asp,
                interpolation=None,
                cmap=cmap,
                clim=(self.__window[0], self.__window[0] + self.__window[1]),
            )
            ax_sag.imshow(
                nda.__getitem__(s_sag),
                origin="lower",
                aspect=asp,
                interpolation=None,
                cmap=cmap,
                clim=(self.__window[0], self.__window[0] + self.__window[1]),
            )

            ax_ax.axis("off")
            ax_cor.axis("off")
            ax_sag.axis("off")

            self.__figure.subplots_adjust(
                left=0, right=1, wspace=0.01, hspace=0.01, top=1, bottom=0
            )

        else:
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
            ax.imshow(
                nda.__getitem__(s),
                aspect=asp,
                interpolation=None,
                origin=org,
                cmap=cmap,
                clim=(self.__window[0], self.__window[0] + self.__window[1]),
            )
            ax.axis("off")

            self.__figure.subplots_adjust(left=0, right=1, bottom=0, top=1)

    def overlay_comparison(self):
        """Display an overlay comparison

        Args:
            color_rotation (float, optional): The hue used to color the original image (0 - 0.5).
        """

        if len(self.__comparison_overlays)>1:
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

            s_ax = self.return_slice("z", self.__cut[0])
            s_cor = self.return_slice("y", self.__cut[1])
            s_sag = self.return_slice("x", self.__cut[2])

            nda_a = nda_original.__getitem__(s_ax)
            nda_b = nda_new.__getitem__(s_ax)

            nda_a_norm = (np.clip(nda_a, window[0], window[1])-window[0])/(window[1]-window[0])
            nda_b_norm = (np.clip(nda_b, window[0], window[1])-window[0])/(window[1]-window[0])

            nda_colour = np.stack([color_rotation*(nda_a_norm>nda_b_norm) + (0.5+color_rotation)*(nda_a_norm<=nda_b_norm),
                                np.abs(nda_a_norm - nda_b_norm),
                                (nda_a_norm + nda_b_norm)/2],
                                axis=-1)

            ax_ax.imshow(
                hsv2rgb(nda_colour),
                aspect=1.0,
                interpolation=None,
            )

            nda_a = nda_original.__getitem__(s_cor)
            nda_b = nda_new.__getitem__(s_cor)

            nda_a_norm = (np.clip(nda_a, window[0], window[1])-window[0])/(window[1]-window[0])
            nda_b_norm = (np.clip(nda_b, window[0], window[1])-window[0])/(window[1]-window[0])

            nda_colour = np.stack([color_rotation*(nda_a_norm>nda_b_norm) + (0.5+color_rotation)*(nda_a_norm<=nda_b_norm),
                                np.abs(nda_a_norm - nda_b_norm),
                                (nda_a_norm + nda_b_norm)/2],
                                axis=-1)

            ax_cor.imshow(
                hsv2rgb(nda_colour),
                origin="lower",
                aspect=asp,
                interpolation=None,
            )

            nda_a = nda_original.__getitem__(s_sag)
            nda_b = nda_new.__getitem__(s_sag)

            nda_a_norm = (np.clip(nda_a, window[0], window[1])-window[0])/(window[1]-window[0])
            nda_b_norm = (np.clip(nda_b, window[0], window[1])-window[0])/(window[1]-window[0])

            nda_colour = np.stack([color_rotation*(nda_a_norm>nda_b_norm) + (0.5+color_rotation)*(nda_a_norm<=nda_b_norm),
                                np.abs(nda_a_norm - nda_b_norm),
                                (nda_a_norm + nda_b_norm)/2],
                                axis=-1)

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

            nda_a_norm = (np.clip(nda_a, window[0], window[1])-window[0])/(window[1]-window[0])
            nda_b_norm = (np.clip(nda_b, window[0], window[1])-window[0])/(window[1]-window[0])

            nda_colour = np.stack([color_rotation*(nda_a_norm>nda_b_norm) + (0.5+color_rotation)*(nda_a_norm<=nda_b_norm),
                                np.abs(nda_a_norm - nda_b_norm),
                                (nda_a_norm + nda_b_norm)/2],
                                axis=-1)

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
        """

        limits = self.__limits

        if limits is not None:
            if self.__axis == "ortho":
                ax_ax, _, ax_cor, ax_sag = self.__figure.axes
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

                ratio_x = ( (cor_1 - cor_0) + (sag_1 - sag_0) ) / ( (cor_orig_1 - cor_orig_0) + (sag_orig_1 - sag_orig_0) )
                ratio_y = ( 1/asp*(cor_1 - cor_0) + (ax_1 - ax_0) ) / ( 1/asp*(cor_orig_1 - cor_orig_0) + (ax_orig_1 - ax_orig_0) )

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
                    width_ratios=[sag_size, cor_size]
                )

                ax_ax.set_position(gs[0].get_position(self.__figure))
                ax_ax.set_subplotspec(gs[0]) 

                ax_cor.set_position(gs[2].get_position(self.__figure))
                ax_cor.set_subplotspec(gs[2]) 

                ax_sag.set_position(gs[3].get_position(self.__figure))
                ax_sag.set_subplotspec(gs[3]) 

                fig_size_x, fig_size_y = self.__figure.get_size_inches()
                fig_size_y = fig_size_y*ratio_y/ratio_x
                
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

                ratio_x = (np.abs(x_1 - x_0)/np.abs(x_orig_1 - x_orig_0))
                ratio_y = (np.abs(y_1 - y_0)/np.abs(y_orig_1 - y_orig_0))

                ax.set_xlim(x_0,x_1)
                ax.set_ylim(y_0,y_1)  

                fig_size_x, fig_size_y = self.__figure.get_size_inches()
                fig_size_y = fig_size_y*ratio_y/ratio_x
                
                self.__figure.set_size_inches(fig_size_x, fig_size_y)


    def overlay_contours(self):
        """Overlay the contours on to the current figure image
        """

        plot_dict = {
            contour.name: sitk.GetArrayFromImage(contour.image) for contour in self.__contours
        }
        color_map = self.__contour_color_base(np.linspace(0, 1, len(self.__contours)))
        colors = [
            color_map[i] if contour.color is None else contour.color
            for i, contour in enumerate(self.__contours)
        ]

        # Test types of axes
        axes = self.__figure.axes
        if self.__axis in ["x", "y", "z"]:
            ax = axes[0]
            s = self.return_slice(self.__axis, self.__cut)
            for index, c_name in enumerate(plot_dict.keys()):
                try:
                    ax.contour(
                        plot_dict[c_name].__getitem__(s),
                        colors=[colors[index]],
                        levels=[0],
                        # alpha=0.8,
                        linewidths=2,
                        label=c_name,
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
                    linewidths=2,
                    colors=[colors[index]],
                )
                temp.collections[0].set_label(c_name)

                ax_cor.contour(
                    plot_dict[c_name].__getitem__(s_cor),
                    levels=[0],
                    linewidths=2,
                    colors=[colors[index]],
                )
                ax_sag.contour(
                    plot_dict[c_name].__getitem__(s_sag),
                    levels=[0],
                    linewidths=2,
                    colors=[colors[index]],
                )

            if self.__show_legend:
                ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

        else:
            raise ValueError('Axis is must be one of "x","y","z","ortho".')


    def overlay_scalar_field(self):
        """Overlay the scalar image onto the existing figure
        """

        for scalar in self.__scalar_overlays:

            alpha = scalar.alpha
            sMin = scalar.min_value

            scalar_image = scalar.image
            nda = sitk.GetArrayFromImage(scalar_image)
            nda = nda / nda.max()
            nda = np.ma.masked_where(nda < sMin, nda)

            sp_plane, _, sp_slice = scalar_image.GetSpacing()
            asp = (1.0 * sp_slice) / sp_plane

            # Test types of axes
            axes = self.__figure.axes
            if len(axes) == 1:
                ax = axes[0]
                s = self.return_slice(self.__axis, self.__cut)
                sp = ax.imshow(
                    nda.__getitem__(s),
                    interpolation=None,
                    cmap=scalar.colormap,
                    clim=(0, 1),
                    aspect={"z": 1, "y": asp, "x": asp}[self.__axis],
                    origin={"z": "upper", "y": "lower", "x": "lower"}[self.__axis],
                    vmin=sMin,
                    alpha=alpha,
                )

                if self.__show_colorbar:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = self.__figure.colorbar(sp, cax=cax, orientation="vertical")
                    cbar.set_label("Probability", fontsize=16)

                    fX, fY = self.__figure.get_size_inches()
                    self.__figure.set_size_inches(fX * 1.15, fY)
                    self.__figure.subplots_adjust(left=0, right=0.88, bottom=0, top=1)

            elif len(axes) == 4:
                ax_ax, _, ax_cor, ax_sag = axes

                sAx = self.return_slice("z", self.__cut[0])
                sCor = self.return_slice("y", self.__cut[1])
                sSag = self.return_slice("x", self.__cut[2])

                ax_ax.imshow(
                    nda.__getitem__(sAx),
                    interpolation=None,
                    cmap=scalar.colormap,
                    clim=(0, 1),
                    aspect=1,
                    vmin=sMin,
                    alpha=alpha,
                )

                ax_cor.imshow(
                    nda.__getitem__(sCor),
                    interpolation=None,
                    cmap=scalar.colormap,
                    clim=(0, 1),
                    origin="lower",
                    aspect=asp,
                    vmin=sMin,
                    alpha=alpha,
                )

                ax_sag.imshow(
                    nda.__getitem__(sSag),
                    interpolation=None,
                    cmap=scalar.colormap,
                    clim=(0, 1),
                    origin="lower",
                    aspect=asp,
                    vmin=sMin,
                    alpha=alpha,
                )

    def overlay_vector_field(self):
        """Overlay vector field onto existing figure
        
        """
        for vector in self.__vector_overlays:

            image = vector.image 
            name = vector.name 
            colormap = vector.colormap 
            alpha = vector.alpha 
            arrow_scale = vector.arrow_scale 
            arrow_width = vector.arrow_width 
            subsample = vector.subsample 
            color_function = vector.color_function 

            inverse_vector_image = image #sitk.InvertDisplacementField(image)
            vector_nda = sitk.GetArrayFromImage(inverse_vector_image)

            sp_plane, _, sp_slice = image.GetSpacing()
            asp = (1.0 * sp_slice) / sp_plane

            # Test types of axes
            axes = self.__figure.axes
            if len(axes) == 1:
                ax = axes[0]

                if hasattr(subsample, "__iter__"):
                    raise ValueError("You have selected an iterable subsampling factor for a\
                                      single axis. Behaviour undefined in this situation.")

                slicer = self.subsample_vector_field(self.__axis, self.__cut, subsample)
                vector_nda_slice = vector_nda.__getitem__(slicer)

                vector_ax = vector_nda_slice[:,:,2].T
                vector_cor = vector_nda_slice[:,:,1].T
                vector_sag = vector_nda_slice[:,:,0].T

                vector_plot_x, vector_plot_y, vector_plot_z = self.reorientate_vector_field(
                    self.__axis,
                    vector_ax,
                    vector_cor,
                    vector_sag
                )

                plot_x_loc, plot_y_loc = self.vector_image_grid(self.__axis, vector_nda, subsample)

                if color_function == 'perpendicular':
                    vector_color = vector_plot_z
                elif color_function == 'magnitude':
                    vector_color = np.sqrt(vector_plot_x**2 + vector_plot_y**2 + vector_plot_z**2)

                ax.quiver(
                    plot_x_loc,
                    plot_y_loc,
                    vector_plot_x,
                    vector_plot_y,
                    vector_color,
                    cmap=colormap,
                    units='xy',
                    scale=1/arrow_scale,
                    width=arrow_width,
                    minlength=0,
                    linewidth=1
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

                if hasattr(subsample, "__iter__"):
                    subsample_ax, subsample_cor, subsample_sag = subsample
                else:
                    subsample_ax, subsample_cor, subsample_sag = (subsample,)*3

                for plot_axes, im_axis, im_cut  in zip(
                    (ax_ax, ax_cor, ax_sag),
                    ("z","y","x"),
                    self.__cut
                ):

                    slicer = self.subsample_vector_field(im_axis, im_cut, subsample)
                    vector_nda_slice = vector_nda.__getitem__(slicer)

                    vector_ax = vector_nda_slice[:,:,2].T
                    vector_cor = vector_nda_slice[:,:,1].T
                    vector_sag = vector_nda_slice[:,:,0].T

                    vector_plot_x, vector_plot_y, vector_plot_z = self.reorientate_vector_field(
                        im_axis,
                        vector_ax,
                        vector_cor,
                        vector_sag
                    )

                    plot_x_loc, plot_y_loc = self.vector_image_grid(im_axis, vector_nda, subsample)

                    if color_function == 'perpendicular':
                        vector_color = vector_plot_z
                    elif color_function == 'magnitude':
                        vector_color = np.sqrt(vector_plot_x**2 + vector_plot_y**2 + vector_plot_z**2)


                    sp = plot_axes.quiver(
                        plot_x_loc,
                        plot_y_loc,
                        vector_plot_x,
                        vector_plot_y,
                        vector_color,
                        cmap=colormap,
                        units='xy',
                        scale=1/arrow_scale,
                        width=arrow_width,
                        minlength=0,
                        linewidth=1
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
            axes = self.__figure.axes
            if len(axes) == 1:
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
