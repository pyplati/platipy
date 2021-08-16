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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable  # , AxesGrid, ImageGrid

import numpy as np
import SimpleITK as sitk

from loguru import logger


class View:
    """
    Class to display a single 2D view comprising images, contours, vectors, etc
    """

    def __init__(
        self,
        image,
        axis=None,
        contours=None,
        bounding_boxes=None,
        scalar_overlays=None,
        vector_overlays=None,
        crosshair_overlay=None
    ):

    if isinstance(image,)
    self.__arr_img = sitk.GetArrayFromImage
    self.__dimension = self.__arr_img.ndim

    def create_slice(
        self,
        axis_slice=None,
    ):
        """Set the slice (location) at which to cut the image/contours/overlays.
        This should be called first to put all the visuals onto an axis.

        Args:
            slice (int, optional): The slice location. Defaults to None.
        """

        ax = self.__axis
        aspect

        if arr_self.__arr_img.ndim == 2:
            logger.info("Cannot set a slice on a 2D image, ignoring.")
            
            axis_slicer = return_slice()

        if axis_slice is None:
            if arr_self.__arr_img.ndim == 2:
                axis_slice = 

        # First: display the image on the axis

        image = self.__image

        ax.imshow(
            arr_slice,
            aspect=aspect
        )

    def update_slice(
        self,
        slice=None
    ):
