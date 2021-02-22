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


"""
Module name: Visualisation
Author:      Robert Finnegan
Date:        December 2018
Description:
---------------------------------
- Medical Imaging
- AutoSegmentation Output: deformation fields, delineations
- Slice generation
---------------------------------

"""

from __future__ import print_function
import os, sys

import SimpleITK as sitk
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable, AxesGrid, ImageGrid

from matplotlib.colors import hsv_to_rgb
from skimage.color import hsv2rgb
import matplotlib.patches as mpatches


def returnSlice(axis, index):
    if axis == "x":
        s = (slice(None), slice(None), index)
    if axis == "y":
        s = (slice(None), index, slice(None))
    if axis == "z":
        s = (index, slice(None), slice(None))

    return s


def returnVectorFieldSlice(axis, index):
    if axis == "z":
        s = (slice(None), slice(None), index)
    if axis == "y":
        s = (slice(None), index, slice(None))
    if axis == "x":
        s = (index, slice(None), slice(None))
    return s


def returnVectorSlice(skipStep):
    return (slice(None, None, skipStep), slice(None, None, skipStep))


def resampleVF(axis, index, skipStep):
    if axis == "x":
        s = (
            slice(None, None, None),
            slice(None, None, skipStep),
            slice(None, None, skipStep),
        )
    if axis == "y":
        s = (
            slice(None, None, skipStep),
            slice(None, None, None),
            slice(None, None, skipStep),
        )
    if axis == "z":
        s = (
            slice(None, None, skipStep),
            slice(None, None, skipStep),
            slice(None, None, None),
        )
    return s


def vectorImageGrid(axis, vf):
    if axis == "x":
        return np.mgrid[0 : vf.shape[1] : 1, 0 : vf.shape[0] : 1]
    if axis == "y":
        return np.mgrid[0 : vf.shape[2] : 1, 0 : vf.shape[0] : 1]
    if axis == "z":
        return np.mgrid[0 : vf.shape[2] : 1, 0 : vf.shape[1] : 1]


def reorientateVectorField(axis, u, v, w):
    if axis == "x":
        return 1.0 * v, 1.0 * w, -1.0 * u
    if axis == "y":
        return -1.0 * u, -1.0 * w, v
    if axis == "z":
        return -1.0 * u, v, -1.0 * w


def keepLargestConnectedComponent(binaryImage):
    componentLabelledImage = sitk.ConnectedComponent(binaryImage)
    labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
    labelShapeFilter.Execute(componentLabelledImage)
    labels = labelShapeFilter.GetLabels()
    largestLabel = max(labels, key=labelShapeFilter.GetNumberOfPixels)
    return sitk.Equal(componentLabelledImage, largestLabel)


def probabilityThreshold(inputImage, lowerThreshold=0.5):
    runningImage = sitk.BinaryThreshold(
        inputImage, lowerThreshold=lowerThreshold, upperThreshold=1.0
    )
    runningImage = sitk.BinaryFillhole(runningImage)
    return keepLargestConnectedComponent(runningImage)


def displaySlice(
    im,
    axis="ortho",
    cut=None,
    figSize=6,
    cmap=plt.cm.Greys_r,
    window=[-250, 500],
    addCBar=False,
    cmap_label="",
    origin="normal",
):
    if type(im) == sitk.Image:
        nda = sitk.GetArrayFromImage(im)
    else:
        print("Image type not recognised, must be SimpleITK image format.")

    (AxSize, CorSize, SagSize) = nda.shape
    spPlane, _, spSlice = im.GetSpacing()
    asp = (1.0 * spSlice) / spPlane

    if axis == "ortho":
        fSize = (
            figSize,
            figSize * (asp * AxSize + CorSize) / (1.0 * SagSize + CorSize),
        )

        fig, ((axAx, blank), (axCor, axSag)) = plt.subplots(
            2,
            2,
            figsize=fSize,
            gridspec_kw={
                "height_ratios": [(CorSize) / (asp * AxSize), 1],
                "width_ratios": [SagSize, CorSize],
            },
        )
        blank.axis("off")

        if cut is None:
            sliceAx = int(AxSize / 2.0)
            sliceCor = int(CorSize / 2.0)
            sliceSag = int(SagSize / 2.0)

            cut = [sliceAx, sliceCor, sliceSag]

        sAx = returnSlice("z", cut[0])
        sCor = returnSlice("y", cut[1])
        sSag = returnSlice("x", cut[2])

        imAx = axAx.imshow(
            nda.__getitem__(sAx),
            aspect=1.0,
            interpolation=None,
            origin={'normal':'upper','reversed':'lower'}[origin]
            cmap=cmap,
            clim=(window[0], window[0] + window[1]),
        )
        imCor = axCor.imshow(
            nda.__getitem__(sCor),
            origin={'normal':'lower', 'reversed':'upper'}[origin],
            aspect=asp,
            interpolation=None,
            cmap=cmap,
            clim=(window[0], window[0] + window[1]),
        )
        imSag = axSag.imshow(
            nda.__getitem__(sSag),
            origin={'normal':'lower', 'reversed':'upper'}[origin],
            aspect=asp,
            interpolation=None,
            cmap=cmap,
            clim=(window[0], window[0] + window[1]),
        )

        axAx.axis("off")
        axCor.axis("off")
        axSag.axis("off")

        if addCBar:
            divider = make_axes_locatable(axAx)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(imAx, cax=cax, orientation="vertical")
            cbar.set_label(cmap_label)

        fig.subplots_adjust(left=0, right=1, wspace=0.01, hspace=0.01, top=1, bottom=0)

    else:
        if axis == "x" or axis == "sag":
            fSize = (figSize, figSize * (asp * AxSize) / (1.0 * CorSize))
            fig, ax = plt.subplots(1, 1, figsize=(fSize))
            org = {'normal':'lower', 'reversed':'upper'}[origin]
            if not cut:
                cut = int(SagSize / 2.0)

        if axis == "y" or axis == "cor":
            fSize = (figSize, figSize * (asp * AxSize) / (1.0 * SagSize))
            fig, ax = plt.subplots(1, 1, figsize=(fSize))
            org = {'normal':'lower', 'reversed':'upper'}[origin]
            if not cut:
                cut = int(CorSize / 2.0)

        if axis == "z" or axis == "ax":
            asp = 1
            fSize = (figSize, figSize * (asp * CorSize) / (1.0 * SagSize))
            fig, ax = plt.subplots(1, 1, figsize=(fSize))
            org = {'normal':'upper','reversed':'lower'}[origin]
            if not cut:
                cut = int(AxSize / 2.0)

        s = returnSlice(axis, cut)
        ax.imshow(
            nda.__getitem__(s),
            aspect=asp,
            interpolation=None,
            origin=org,
            cmap=cmap,
            clim=(window[0], window[0] + window[1]),
        )
        ax.axis("off")

        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    return fig, axis, cut


def displayComparisonSlice(
    im_a,
    im_b,
    axis="ortho",
    cut=None,
    figSize=6,
    window=[-250, 500],
    addCBar=False,
    colour_rotation=0.35,
    cmap_label="",
):
    if type(im_a) == sitk.Image:
        nda_a = sitk.GetArrayFromImage(im_a)
    else:
        print("Image type not recognised, must be SimpleITK image format.")

    if type(im_b) == sitk.Image:
        nda_b = sitk.GetArrayFromImage(im_b)
    else:
        print("Image type not recognised, must be SimpleITK image format.")

    if im_a.GetSize() != im_b.GetSize():
        raise ValueError("Images must be the same size for this function!")

    (AxSize, CorSize, SagSize) = nda_a.shape
    spPlane, _, spSlice = im_a.GetSpacing()
    asp = (1.0 * spSlice) / spPlane

    nda_a_norm = (np.clip(nda_a, window[0], window[1]) - window[0]) / (
        window[1] - window[0]
    )
    nda_b_norm = (np.clip(nda_b, window[0], window[1]) - window[0]) / (
        window[1] - window[0]
    )

    nda_colour = np.stack(
        [
            colour_rotation * (nda_a_norm > nda_b_norm)
            + (0.5 + colour_rotation) * (nda_a_norm <= nda_b_norm),
            np.abs(nda_a_norm - nda_b_norm),
            (nda_a_norm + nda_b_norm) / 2,
        ],
        axis=-1,
    )

    if axis == "ortho":
        fSize = (
            figSize,
            figSize * (asp * AxSize + CorSize) / (1.0 * SagSize + CorSize),
        )

        fig, ((axAx, blank), (axCor, axSag)) = plt.subplots(
            2,
            2,
            figsize=fSize,
            gridspec_kw={
                "height_ratios": [(CorSize) / (asp * AxSize), 1],
                "width_ratios": [SagSize, CorSize],
            },
        )
        blank.axis("off")

        if cut is None:
            sliceAx = int(AxSize / 2.0)
            sliceCor = int(CorSize / 2.0)
            sliceSag = int(SagSize / 2.0)

            cut = [sliceAx, sliceCor, sliceSag]

        sAx = returnSlice("z", cut[0])
        sCor = returnSlice("y", cut[1])
        sSag = returnSlice("x", cut[2])

        nda_colour_ax = hsv2rgb(nda_colour.__getitem__(sAx))
        nda_colour_cor = hsv2rgb(nda_colour.__getitem__(sCor))
        nda_colour_sag = hsv2rgb(nda_colour.__getitem__(sSag))

        imAx = axAx.imshow(nda_colour_ax, aspect=1.0, interpolation=None)
        imCor = axCor.imshow(
            nda_colour_cor, origin="lower", aspect=asp, interpolation=None
        )
        imSag = axSag.imshow(
            nda_colour_sag, origin="lower", aspect=asp, interpolation=None
        )

        axAx.axis("off")
        axCor.axis("off")
        axSag.axis("off")

        if addCBar:
            divider = make_axes_locatable(axAx)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(imAx, cax=cax, orientation="vertical")
            cbar.set_label(cmap_label)

        fig.subplots_adjust(left=0, right=1, wspace=0.01, hspace=0.01, top=1, bottom=0)

    else:
        if axis == "x" or axis == "sag":
            fSize = (figSize, figSize * (asp * AxSize) / (1.0 * CorSize))
            fig, ax = plt.subplots(1, 1, figsize=(fSize))
            org = "lower"
            if not cut:
                cut = int(SagSize / 2.0)

        if axis == "y" or axis == "cor":
            fSize = (figSize, figSize * (asp * AxSize) / (1.0 * SagSize))
            fig, ax = plt.subplots(1, 1, figsize=(fSize))
            org = "lower"
            if not cut:
                cut = int(CorSize / 2.0)

        if axis == "z" or axis == "ax":
            asp = 1
            fSize = (figSize, figSize * (asp * CorSize) / (1.0 * SagSize))
            fig, ax = plt.subplots(1, 1, figsize=(fSize))
            org = "upper"
            if not cut:
                cut = int(AxSize / 2.0)

        s = returnSlice(axis, cut)
        nda_colour_slice = hsv2rgb(nda_colour.__getitem__(s))
        ax.imshow(nda_colour_slice, aspect=asp, interpolation=None, origin=org)
        ax.axis("off")

        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    return fig, axis, cut


def overlayContour(
    contour_input, fig, axis, cut, use_legend=True, color_base=plt.cm.Blues
):

    # Check input type
    if type(contour_input) == sitk.Image:
        use_legend = False
        plot_dict = {"input": sitk.GetArrayFromImage(contour_input)}
        color = color_base(126)

    elif type(contour_input) == dict:
        if all(map(lambda i: type(i) == sitk.Image, contour_input.values())):
            plot_dict = {i: sitk.GetArrayFromImage(j) for i, j in contour_input.items()}
            colors = color_base(np.linspace(0, 1, len(contour_input.keys())))

        else:
            raise ValueError(
                "If passing a dictionary, all values must be of type SimpleITK.Image"
            )

    else:
        raise ValueError(
            "Input not recognised, this must be either a single (or dictionary of) SimpleITK.Image"
        )

    # Test types of axes
    axes = fig.axes[:4]
    if axis in ["x", "y", "z"]:
        ax = axes[0]
        s = returnSlice(axis, cut)
        for index, c_name in enumerate(plot_dict.keys()):
            try:
                ax.contour(
                    plot_dict[c_name].__getitem__(s),
                    colors=colors[index],
                    levels=[0],
                    # alpha=0.8,
                    linewidths=1,
                    label=c_name,
                )
            except:
                None

    elif axis == "ortho":
        axAx, _, axCor, axSag = axes

        ax = axAx

        sAx = returnSlice("z", cut[0])
        sCor = returnSlice("y", cut[1])
        sSag = returnSlice("x", cut[2])

        for index, c_name in enumerate(plot_dict.keys()):

            temp = axAx.contour(
                plot_dict[c_name].__getitem__(sAx),
                levels=[0],
                linewidths=2,
                colors=[colors[index]],
            )
            temp.collections[0].set_label(c_name)

            axCor.contour(
                plot_dict[c_name].__getitem__(sCor),
                levels=[0],
                linewidths=2,
                colors=[colors[index]],
            )
            axSag.contour(
                plot_dict[c_name].__getitem__(sSag),
                levels=[0],
                linewidths=2,
                colors=[colors[index]],
            )

    else:
        raise ValueError('Axis is must be one of "x","y","z","ortho".')

    if use_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))

    return fig


def overlayScalarField(
    scalarIm,
    fig,
    axis,
    cut,
    color_base=plt.cm.Spectral,
    addCBar=False,
    window=[0, 1],
    plotArgs={"alpha": 0.75, "sMin": 0.01},
):

    alpha = plotArgs["alpha"]
    sMin = plotArgs["sMin"]

    nda = sitk.GetArrayFromImage(scalarIm)
    nda = np.ma.masked_where(nda < sMin, nda)

    (AxSize, CorSize, SagSize) = nda.shape
    spPlane, _, spSlice = scalarIm.GetSpacing()
    asp = (1.0 * spSlice) / spPlane

    # Test types of axes
    axes = fig.axes
    if len(axes) == 1:
        ax = axes[0]
        s = returnSlice(axis, cut)
        sp = ax.imshow(
            nda.__getitem__(s),
            interpolation=None,
            cmap=color_base,
            clim=(window[0], window[0] + window[1]),
            aspect={"z": 1, "y": asp, "x": asp}[axis],
            origin={"z": "upper", "y": "lower", "x": "lower"}[axis],
            vmin=sMin,
            alpha=alpha,
        )

        if addCBar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(sp, cax=cax, orientation="vertical")
            cbar.set_label("Probability", fontsize=16)

            fX, fY = fig.get_size_inches()
            fig.set_size_inches(fX * 1.15, fY)
            fig.subplots_adjust(left=0, right=0.88, bottom=0, top=1)

    elif len(axes) == 4:
        axAx, blank, axCor, axSag = axes

        sAx = returnSlice("z", cut[0])
        sCor = returnSlice("y", cut[1])
        sSag = returnSlice("x", cut[2])

        sp = axAx.imshow(
            nda.__getitem__(sAx),
            interpolation=None,
            cmap=color_base,
            clim=(window[0], window[0] + window[1]),
            aspect=1,
            vmin=sMin,
            alpha=alpha,
        )

        axCor.imshow(
            nda.__getitem__(sCor),
            interpolation=None,
            cmap=color_base,
            clim=(window[0], window[0] + window[1]),
            origin="lower",
            aspect=asp,
            vmin=sMin,
            alpha=alpha,
        )

        axSag.imshow(
            nda.__getitem__(sSag),
            interpolation=None,
            cmap=color_base,
            clim=(window[0], window[0] + window[1]),
            origin="lower",
            aspect=asp,
            vmin=sMin,
            alpha=alpha,
        )

        if addCBar:
            divider = make_axes_locatable(axAx)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(sp, cax=cax, orientation="vertical")
            cbar.set_label("", fontsize=16)

    return fig


# def overlayVectorField(vectorIm, fig, axis, cut, color_base=plt.cm.Spectral, addCBar=False, plotArgs={'alpha':0.75, 'arrow_scale':0.25, 'arrow_width':1, 'skip_step'=}):

#     alpha = plotArgs['alpha']
#     arrow_scale = plotArgs['arrow_scale']
#     arrow_width = plotArgs['arrow_width']

#     nda = sitk.GetArrayFromImage(vectorIm)

#     u = sitk.GetArrayFromImage(sitk.VectorIndexSelectionCast(vectorImage, 0)) # x-component = u
#     v = sitk.GetArrayFromImage(sitk.VectorIndexSelectionCast(vectorImage, 1)) # y-component = v
#     w = sitk.GetArrayFromImage(sitk.VectorIndexSelectionCast(vectorImage, 2)) # z-component = w

#     # Test types of axes
#     axes = fig.axes
#     if len(axes) == 1:
#         ax = axes[0]

#         sV = returnVectorSlice(skip_step)
#         plot_x_arr, plot_y_arr = vectorImageGrid(axis, nda)

#         plot_x, plot_y = plot_x_arr.__getitem__(sV), plot_y_arr.__getitem__(sV)

#         sV3D = resampleVF(axis, cut, skip_step)
#         c,d,e = u.T.__getitem__(sV3D), v.T.__getitem__(sV3D), w.T.__getitem__(sV3D)

#         sVF = returnVectorFieldSlice(axis,cut)
#         plot_u,plot_v,plot_w = reorientateVectorField(axis,c,d,e)


#         vfQ = ax.quiver(plot_x,
#                         plot_y,
#                         plot_u.__getitem__(sVF),
#                         plot_v.__getitem__(sVF),
#                         plot_w.__getitem__(sVF),
#                         cmap=plt.cm.Spectral,
#                         units='xy',
#                         scale=arrow_scale,
#                         width=arrow_width
#                         )


#         if addCBar:
#             divider = make_axes_locatable(ax)
#             cax = divider.append_axes('right', size='5%', pad=0.05)
#             cbar = fig.colorbar(sp, cax=cax, orientation='vertical')
#             cbar.set_label('Probability', fontsize=16)

#             fX, fY = fig.get_size_inches()
#             fig.set_size_inches(fX*1.15, fY)
#             fig.subplots_adjust(left=0, right=0.88, bottom=0, top=1)

#     elif len(axes) == 4:
#         axAx, blank, axCor, axSag = axes

#         sAx = returnSlice("z", cut[0])
#         sCor = returnSlice("y", cut[1])
#         sSag = returnSlice("x", cut[2])

#         axAx.imshow(
#                 nda.__getitem__(sAx),
#                 interpolation=None,
#                 cmap=color_base,
#                 clim=(0,1),
#                 aspect=1,
#                 vmin=sMin,
#                 alpha=alpha
#                 )

#         axCor.imshow(
#                 nda.__getitem__(sCor),
#                 interpolation=None,
#                 cmap=color_base,
#                 clim=(0,1),
#                 origin='lower',
#                 aspect=asp,
#                 vmin=sMin,
#                 alpha=alpha
#                 )

#         axSag.imshow(
#                 nda.__getitem__(sSag),
#                 interpolation=None,
#                 cmap=color_base,
#                 clim=(0,1),
#                 origin='lower',
#                 aspect=asp,
#                 vmin=sMin,
#                 alpha=alpha
#                 )

#     return fig


def overlayBox(box, fig, axis, color="r"):

    # Test types of contours
    if len(box) != 6 or any([type(i) != int for i in box]):
        print("Box not formatted correctly.")
        return None

    else:
        sag0, cor0, ax0, sagD, corD, axD = box

    # Test types of axes
    axes = fig.axes
    if len(axes) == 1:
        ax = axes[0]

        if axis == "z" or axis == "ax":
            axAx.plot(
                [sag0, sag0, sag0 + sagD, sag0 + sagD, sag0],
                [cor0, cor0 + corD, cor0 + corD, cor0, cor0],
                lw=2,
                c=color,
            )
        if axis == "y" or axis == "cor":
            axCor.plot(
                [sag0, sag0 + sagD, sag0 + sagD, sag0, sag0],
                [ax0, ax0, ax0 + axD, ax0 + axD, ax0],
                lw=2,
                c=color,
            )
        if axis == "x" or axis == "sag":
            axSag.plot(
                [cor0, cor0 + corD, cor0 + corD, cor0, cor0],
                [ax0, ax0, ax0 + axD, ax0 + axD, ax0],
                lw=2,
                c=color,
            )

    elif len(axes) == 4:
        axAx, blank, axCor, axSag = axes

        axAx.plot(
            [sag0, sag0, sag0 + sagD, sag0 + sagD, sag0],
            [cor0, cor0 + corD, cor0 + corD, cor0, cor0],
            lw=2,
            c=color,
        )
        axCor.plot(
            [sag0, sag0 + sagD, sag0 + sagD, sag0, sag0],
            [ax0, ax0, ax0 + axD, ax0 + axD, ax0],
            lw=2,
            c=color,
        )
        axSag.plot(
            [cor0, cor0 + corD, cor0 + corD, cor0, cor0],
            [ax0, ax0, ax0 + axD, ax0 + axD, ax0],
            lw=2,
            c=color,
        )

    return fig


def main(arguments):
    return True


if __name__ == "__main__":
    main()
