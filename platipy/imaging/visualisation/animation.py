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

import tempfile
import pathlib
import shutil

import imageio

import numpy as np
import SimpleITK as sitk

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FileMovieWriter


class FileWriter(FileMovieWriter):
    """Class to write image frames for animation"""

    supported_formats = ["png"]

    # pylint: disable=arguments-differ, attribute-defined-outside-init
    def setup(self, fig, dpi, frame_prefix):
        super().setup(fig, dpi, frame_prefix)
        self.fname_format_str = "%s%%d.%s"
        self.temp_prefix, self.frame_format = self.outfile.split(".")

    def grab_frame(self, **savefig_kwargs):

        with self._frame_sink() as myframesink:
            self.fig.savefig(myframesink, format="png", dpi=self.dpi, **savefig_kwargs)

    def finish(self):
        self._frame_sink().close()


def generate_animation_from_image_sequence(
    image_list,
    output_file="animation.gif",
    fps=10,
    contour_list=None,
    scalar_list=None,
    figure_size_in=6,
    image_cmap=plt.cm.get_cmap("Greys_r"),
    contour_cmap=plt.cm.get_cmap("jet"),
    scalar_cmap=plt.cm.get_cmap("magma"),
    image_window=[-1000, 800],
    scalar_min=None,
    scalar_max=None,
    scalar_alpha=0.5,
    image_origin="lower",
):
    """Make a GIF animation from the list of images supplied

    Args:
        image_list (list): List of 2D sitk.Image's to render
        output_file (str, optional): Path to output GIF file. Defaults to "animation.gif".
        fps (int, optional): Frames per second. Defaults to 10.
        contour_list (list, optional): List of contours to overlay. Defaults to None.
        scalar_list (list, optional): List of scalars to overlay. Defaults to None.
        figure_size_in (int, optional): Size of figure. Defaults to 6.
        image_cmap ([type], optional): Colormap for image. Defaults to plt.cm.get_cmap("Greys_r").
        contour_cmap ([type], optional): Colormap for contours. Defaults to plt.cm.get_cmap("jet").
        scalar_cmap ([type], optional): Colormap for scalars. Defaults to plt.cm.get_cmap("magma").
        image_window (list, optional): Window for image. Defaults to [-1000, 800].
        scalar_min (float, optional): Minimum value for scalar. Defaults to None.
        scalar_max (float, optional): Maximum value for scalar. Defaults to None.
        scalar_alpha (float, optional): Alpha value for scalar. Defaults to 0.5.
        image_origin (str, optional): Origin of image. Defaults to "lower".

    Raises:
        ValueError: Raised if image list does not contain sitk.Image's

    Returns:
        matplotlib.animation.FuncAnimation: The animation object
    """

    if not isinstance(image_list[0], sitk.Image):
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
    if contour_list:

        if isinstance(contour_list[0], sitk.Image):
            plot_dict = {"_": contour_list[0]}
            contour_labels = False
        else:
            plot_dict = contour_list[0]
            contour_labels = True

        color_map = contour_cmap(np.linspace(0, 1, len(plot_dict)))

        for index, (contour_name, contour) in enumerate(plot_dict.items()):

            display_contours = ax.contour(
                sitk.GetArrayFromImage(contour),
                colors=[color_map[index]],
                levels=[0],
                linewidths=2,
            )

            display_contours.collections[0].set_label(contour_name)

        if contour_labels:
            approx_scaling = figure_size_in / (len(plot_dict.keys()))
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(0.05, 0.95),
                fontsize=min([10, 16 * approx_scaling]),
            )

    if scalar_list:

        if not scalar_min:
            scalar_min = np.min([sitk.GetArrayFromImage(i) for i in scalar_list])
        if not scalar_max:
            scalar_max = np.max([sitk.GetArrayFromImage(i) for i in scalar_list])

        display_scalar = ax.imshow(
            np.ma.masked_outside(sitk.GetArrayFromImage(scalar_list[0]), scalar_min, scalar_max),
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
        if contour_list:
            try:
                ax.collections = []
            except ValueError:
                pass

            if not isinstance(contour_list[i], dict):
                plot_dict = {"_": contour_list[i]}
            else:
                plot_dict = contour_list[i]

            color_map = contour_cmap(np.linspace(0, 1, len(plot_dict)))

            for index, contour in enumerate(plot_dict.values()):

                ax.contour(
                    sitk.GetArrayFromImage(contour),
                    colors=[color_map[index]],
                    levels=[0],
                    linewidths=2,
                )

        if scalar_list:
            nda = sitk.GetArrayFromImage(scalar_list[i])
            display_scalar.set_data(np.ma.masked_outside(nda, scalar_min, scalar_max))

        return (display_image,)

    # create animation using the animate() function with no repeat
    animation_result = animation.FuncAnimation(
        fig,
        animate,
        frames=np.arange(0, len(image_list), 1),
        interval=10,
        blit=True,
        repeat=False,
    )

    # Save animation
    tmp_path = tempfile.mkdtemp()
    animation_result.save(f"{tmp_path}/tmp.format", writer=FileWriter())

    # Save the GIF
    images = []
    for filename in pathlib.Path(tmp_path).glob("tmp*.png"):

        try:
            images.append(imageio.imread(filename))
        except RuntimeError:
            # Skip frames which are corrupt
            pass

    imageio.mimsave(output_file, images, fps=fps)

    # Clean up
    shutil.rmtree(tmp_path)

    return animation_result
