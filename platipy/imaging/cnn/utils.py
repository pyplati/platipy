import numpy as np
import SimpleITK as sitk

from platipy.imaging.label.utils import get_union_mask, get_intersection_mask


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


def preprocess_image(img, spacing=[1, 1, 1], crop_to_mm=128):

    img = sitk.Cast(img, sitk.sitkFloat32)
    img = sitk.IntensityWindowing(
        img, windowMinimum=-500.0, windowMaximum=500.0, outputMinimum=-1.0, outputMaximum=1.0
    )

    new_size = sitk.VectorUInt32(3)
    new_size[0] = int(img.GetSize()[0] * (img.GetSpacing()[0] / spacing[0]))
    new_size[1] = int(img.GetSize()[1] * (img.GetSpacing()[1] / spacing[1]))
    new_size[2] = int(img.GetSize()[2] * (img.GetSpacing()[2] / spacing[2]))

    if crop_to_mm:
        if new_size[0] < crop_to_mm:
            new_size[0] = crop_to_mm

        if new_size[1] < crop_to_mm:
            new_size[1] = crop_to_mm

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

    if crop_to_mm:
        center_x = img.GetSize()[0] / 2
        x_from = int(center_x - crop_to_mm / 2)
        x_to = x_from + crop_to_mm

        center_y = img.GetSize()[1] / 2
        y_from = int(center_y - crop_to_mm / 2)
        y_to = y_from + crop_to_mm

        img = img[x_from:x_to, y_from:y_to, :]

    return img


def resample_mask_to_image(img, mask):

    return sitk.Resample(
        mask,
        img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        mask.GetPixelID(),
    )


def postprocess_mask(pred):

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
    lomif = sitk.LabelOverlapMeasuresImageFilter()
    lomif.Execute(target, pred)
    result["JI"] = lomif.GetJaccardCoefficient()
    result["DSC"] = lomif.GetDiceCoefficient()

    if sitk.GetArrayFromImage(pred).sum() == 0:
        result["HD"] = 1000
        result["ASD"] = 100
    else:
        hdif = sitk.HausdorffDistanceImageFilter()
        hdif.Execute(target, pred)
        result["HD"] = hdif.GetHausdorffDistance()
        result["ASD"] = hdif.GetAverageHausdorffDistance()

    return result
