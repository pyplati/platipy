# Copyright 2021 University of New South Wales, University of Sydney, Ingham Institute

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from platipy.imaging.cnn.localise_net import LocaliseUNet
from platipy.imaging.label.utils import get_union_mask, get_intersection_mask
from platipy.imaging.utils.crop import label_to_roi, crop_to_roi
from platipy.imaging.label.comparison import (
    compute_metric_dsc,
    compute_metric_hd,
    compute_metric_masd,
)


def get_contour_mask(masks, kernel=5):
    """Returns a mask around the region where observer masks don't agree

    Args:
        masks (list): List of observer masks (as sitk.Image)
        kernel (int, optional): The size of the kernal to dilate the contour of. Defaults to 5.

    Returns:
        sitk.Image: The resulting contour mask
    """

    if not hasattr(kernel, "__iter__"):
        kernel = (kernel,) * 3

    union_mask = get_union_mask(masks)
    intersection_mask = get_intersection_mask(masks)

    union_mask = sitk.BinaryDilate(union_mask, np.abs(kernel).astype(int).tolist(), sitk.sitkBall)
    intersection_mask = sitk.BinaryErode(
        intersection_mask, np.abs(kernel).astype(int).tolist(), sitk.sitkBall
    )

    return union_mask - intersection_mask


def preprocess_image(
    img,
    spacing=[1, 1, 1],
    crop_to_grid_size_xy=128,
    intensity_scaling="window",
    intensity_window=[-500, 500],
):
    """Preprocess an image to prepare it for use in a CNN.

    Args:
        img (sitk.Image): [description]
        spacing (list, optional): [description]. Defaults to [1, 1, 1].
        crop_to_grid_size_xy (int|list, optional): Crop to the center grid of this size in x and y
          direction. May be int value which will be use for both x and y size. Or a list containing
          two int values for x and y. Defaults to 128.
        intensity_scaling (str, optional): How to scale the intensity values. Should be one of
        'norm' (center mean and unit variance), 'window' (map window [min max] to [-1 1]), 'none'
        (no intensity scaling applied). Defaults to "window".
        intensity_window (list, optional): List with min and max values to be used when
          intensity_scaling is 'window'. Not used otherwise. Defaults to [-500, 500].

    Returns:
        sitk.Image: The preprocessed image.
    """

    img = sitk.Cast(img, sitk.sitkFloat32)
    if intensity_scaling == "norm":
        img = sitk.Normalize(img)
    elif intensity_scaling == "window":
        img = sitk.IntensityWindowing(
            img,
            windowMinimum=intensity_window[0],
            windowMaximum=intensity_window[1],
            outputMinimum=-1.0,
            outputMaximum=1.0,
        )
    elif intensity_scaling != "none" and intensity_scaling is not None:
        raise ValueError("intensity_scaling should be one of: 'norm', 'window', 'none'")

    new_size = sitk.VectorUInt32(3)
    new_size[0] = int(img.GetSize()[0] * (img.GetSpacing()[0] / spacing[0]))
    new_size[1] = int(img.GetSize()[1] * (img.GetSpacing()[1] / spacing[1]))
    new_size[2] = int(img.GetSize()[2] * (img.GetSpacing()[2] / spacing[2]))

    if crop_to_grid_size_xy:

        if not hasattr(crop_to_grid_size_xy, "__iter__"):
            crop_to_grid_size_xy = (crop_to_grid_size_xy,) * 2

        if new_size[0] < crop_to_grid_size_xy[0]:
            new_size[0] = crop_to_grid_size_xy[0]

        if new_size[1] < crop_to_grid_size_xy[1]:
            new_size[1] = crop_to_grid_size_xy[1]

    img = sitk.Resample(
        img,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        img.GetOrigin(),
        spacing,
        img.GetDirection(),
        -1,
        img.GetPixelID(),
    )

    if crop_to_grid_size_xy:
        center_x = img.GetSize()[0] / 2
        x_from = int(center_x - crop_to_grid_size_xy[0] / 2)
        x_to = x_from + crop_to_grid_size_xy[0]

        center_y = img.GetSize()[1] / 2
        y_from = int(center_y - crop_to_grid_size_xy[1] / 2)
        y_to = y_from + crop_to_grid_size_xy[1]

        img = img[x_from:x_to, y_from:y_to, :]

    return img


def crop_using_localise_model(
    img, localise_model, spacing=[1, 1, 1], crop_to_grid_size=[100, 100, 100]
):
    """Crops an image using a LocaliseUNet

    Args:
        img (SimpleITK.Image): The image to crop
        localise_model (str|Path|LocaliseUNet): The LocaliseUNet or path to checkpoint of
          LocaliseUNet.
        spacing (list, optional): The image spacing (mm) to resample to. Defaults to [1,1,1].
        crop_to_grid_size (list, optional): The size of the grid to crop to. Defaults to
          [100,100,100].

    Returns:
        SimpleITK.Image: The cropped image.
    """

    if isinstance(localise_model, str):
        localise_model = Path(localise_model)

    if isinstance(localise_model, Path):
        if localise_model.is_dir():
            # Find the first actual model checkpoint in this directory
            localise_model = next(localise_model.glob("*.ckpt"))

        localise_model = LocaliseUNet.load_from_checkpoint(localise_model)

    localise_model.eval()
    localise_pred = localise_model.infer(img)

    img = preprocess_image(img, spacing=spacing, crop_to_grid_size_xy=None)
    localise_pred = resample_mask_to_image(img, localise_pred)
    size, index = label_to_roi(localise_pred)

    if not hasattr(crop_to_grid_size, "__iter__"):
        crop_to_grid_size = (crop_to_grid_size,) * 3

    index = [i - int((g - s) / 2) for i, s, g in zip(index, size, crop_to_grid_size)]
    size = crop_to_grid_size
    img_size = img.GetSize()
    for i in range(3):
        if index[i] + size[i] >= img_size[i]:
            index[i] = img_size[i] - size[i] - 1
        if index[i] < 0:
            index[i] = 0

    return crop_to_roi(img, size, index)


def resample_mask_to_image(img, mask):
    """Repsample a mask to the space of the image supplied.

    Args:
        img (sitk.Image): Image to sample to space of.
        mask (sitk.Image): Mask to resample.

    Returns:
        sitk.Image: The resampled mask.
    """

    return sitk.Resample(
        mask,
        img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        mask.GetPixelID(),
    )


def postprocess_mask(pred):
    """Perform postprocessing on a generated auto-segmentation

    Args:
        pred (sitk.Image): The predicted mask

    Returns:
        sitk.Image: The postprocessed mask
    """

    # Take only the largest componenet
    labelled_image = sitk.ConnectedComponent(pred)
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(labelled_image)
    label_indices = label_shape_filter.GetLabels()
    voxel_counts = [label_shape_filter.GetNumberOfPixels(i) for i in label_indices]
    if len(voxel_counts) > 0:
        largest_component_label = label_indices[np.argmax(voxel_counts)]
        largest_component_image = labelled_image == largest_component_label
        pred = sitk.Cast(largest_component_image, sitk.sitkUInt8)

    # Fill any holes in the structure
    pred = sitk.BinaryMorphologicalClosing(pred, (5, 5, 5))
    pred = sitk.BinaryFillhole(pred)

    return pred


def get_metrics(target, pred):

    result = {}
    result["DSC"] = compute_metric_dsc(target, pred)

    target_pixels = sitk.GetArrayFromImage(target).sum()
    pred_pixels = sitk.GetArrayFromImage(pred).sum()

    if pred_pixels == 0 and target_pixels == 0:
        result["HD"] = 0
        result["ASD"] = 0
    elif pred_pixels == 0 or target_pixels == 0:
        result["HD"] = 1000
        result["ASD"] = 100
    else:
        result["HD"] = compute_metric_hd(target, pred)
        result["ASD"] = compute_metric_masd(target, pred)

    return result
