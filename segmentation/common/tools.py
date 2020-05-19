"""
A collection of common tools needed for segmentation tasks
"""

import SimpleITK as sitk


def detect_holes(img, lower_threshold=-10000, upper_threshold=-400):
    """
    Detect all (air) holes in given image. Default threshold values given are defined for CT.

    Returns:
        label_image: image containing labels of all holes detected
        labels: a list of holes detected and some of their properties
    """

    # Threshold to get holes
    btif = sitk.BinaryThresholdImageFilter()
    btif.SetInsideValue(1)
    btif.SetOutsideValue(0)
    btif.SetLowerThreshold(lower_threshold)
    btif.SetUpperThreshold(upper_threshold)
    holes = btif.Execute(img)

    # Get connected components
    ccif = sitk.ConnectedComponentImageFilter()
    label_image = ccif.Execute(holes, False)

    labels = []

    lssif = sitk.LabelShapeStatisticsImageFilter()
    lssif.Execute(label_image)
    for region in range(1, ccif.GetObjectCount()):
        label = {
            "label": region,
            "phys_size": lssif.GetPhysicalSize(region),
            "elongation": lssif.GetElongation(region),
            "roundness": lssif.GetRoundness(region),
            "perimeter": lssif.GetPerimeter(region),
            "flatness": lssif.GetFlatness(region),
        }
        labels.append(label)

    # Sort by size
    labels = sorted(labels, key=lambda i: i["phys_size"], reverse=True)

    return label_image, labels


def get_external_mask(label_image, labels, kernel_radius=5):
    """
    Gets the external mask based on the label image and labels generated using detect_holes
    """

    bmcif = sitk.BinaryMorphologicalClosingImageFilter()
    bmcif.SetKernelType(bmcif.Ball)
    bmcif.SetKernelRadius(kernel_radius)

    # Save the largest as the external contour
    external_mask = sitk.BinaryThreshold(
        label_image, labels[0]["label"], labels[0]["label"]
    )
    external_mask = bmcif.Execute(external_mask)

    return external_mask


def get_lung_mask(label_image, labels, kernel_radius=2):

    lung_idx = 1
    while labels[lung_idx]["flatness"] > 2:
        lung_idx += 1

        if lung_idx > len(labels):
            print("Flatness not satisfied!")
            return None

    bmcif = sitk.BinaryMorphologicalClosingImageFilter()
    bmcif.SetKernelType(bmcif.Ball)
    bmcif.SetKernelRadius(kernel_radius)

    # Save the 2nd largest as the lung mask
    lung_mask = sitk.BinaryThreshold(
        label_image, labels[lung_idx]["label"], labels[lung_idx]["label"]
    )
    lung_mask = bmcif.Execute(lung_mask)

    return lung_mask


def fill_holes(img, label_image, labels, external_mask, lung_mask):
    # Fill all other holes
    img_array = sitk.GetArrayFromImage(img)

    bdif = sitk.BinaryDilateImageFilter()
    bdif.SetKernelType(bdif.Ball)
    bdif.SetKernelRadius(3)

    mask = sitk.BinaryThreshold(label_image, 1, int(img_array.max()))
    mask = sitk.Subtract(mask, external_mask)
    mask = sitk.Subtract(mask, lung_mask)
    mask = bdif.Execute(mask)

    mask_array = sitk.GetArrayFromImage(mask)
    img_array[mask_array == 1] = 50

    filled_img = sitk.GetImageFromArray(img_array)
    filled_img.CopyInformation(img)

    return filled_img


def get_crop_bounding_box(img, mask):

    label_shape_analysis = sitk.LabelShapeStatisticsImageFilter()
    label_shape_analysis.Execute(mask)
    mask_box = label_shape_analysis.GetBoundingBox(True)

    sag_padding = 50
    cor_padding = 50
    ax_padding = 30
    ax_extent = 250

    phys_bb_origin = list(mask.TransformIndexToPhysicalPoint(mask_box[0:3]))
    phys_bb_origin[0] -= sag_padding
    phys_bb_origin[1] -= cor_padding
    phys_bb_origin[2] -= ax_extent - ax_padding
    bb_origin = img.TransformPhysicalPointToIndex(phys_bb_origin)

    phys_bb_size = [0, 0, 0]
    bb_size = [0, 0, 0]
    for i in range(3):
        phys_bb_size[i] = mask_box[3 + i] * mask.GetSpacing()[i]
        if i == 0:
            phys_bb_size[i] += sag_padding * 2
        if i == 1:
            phys_bb_size[i] += cor_padding * 2
        if i == 2:
            phys_bb_size[i] = ax_extent + ax_padding * 2
        bb_size[i] = phys_bb_size[i] / mask.GetSpacing()[i]

    bounding_box = bb_origin + tuple(bb_size)
    bounding_box = [int(i) for i in bounding_box]

    for i in range(3):
        if bounding_box[i] < 0:
            bounding_box[3 + i] = bounding_box[3 + i] + bounding_box[i]
            bounding_box[i] = max(bounding_box[i], 0)

        bounding_box[3 + i] = min(bounding_box[3 + i], img.GetSize()[i] - bounding_box[i])

    return bounding_box
