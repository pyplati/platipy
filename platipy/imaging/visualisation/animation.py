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

import pathlib

import numpy as np
import SimpleITK as sitk

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation


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