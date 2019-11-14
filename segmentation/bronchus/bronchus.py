#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 16:39:27 2018

Bronchus segmentation.  The superior extent of bronchus is around 4cm SUP from Carina (From first
slice where two airways become visible).
In this code I'm using 2cm as it's easier to detect where the airways are getting wider

Areas to improve: parameters could be improved (eg. the median filter, carina detection, etc). The
GenLung_mask is based on old ITK code from a masters student.  I think we can replace this function
by checking the top (sup) slice for an airhole and then connected thresholding.

Code fails on two Liverpool cases:  13 (lungs appear in the sup slice) and 36 (the mask failed to
generate - need to look at this)

@author: Jason Dowling (CSIRO)
"""

import SimpleITK as sitk
from scipy import ndimage

FAST_MODE = True
EXTEND_FROM_CARINA = 20  # ie. 2cm


def fast_mask(img, start, end):
    """Fast masking for area of a 3D volume .

    SimpleITK lacks iterators so voxels need to be set with SetPixel which is horribly slow.
    This code uses numpy arrays to reduce time for one volume from around one minute to about 0.5s

    Args:
        image: Input 3D binary image, start slice for masking (value=0), end slice for masking

    Returns:
        Masked image.  This may be in float, so it might need casting back to original pixel type.
    """
    np_img = sitk.GetArrayFromImage(img).astype(float)
    np_img[start:end, :, :] = 0
    new_img = sitk.GetImageFromArray(np_img)
    new_img.SetSpacing(img.GetSpacing())
    new_img.SetOrigin(img.GetOrigin())
    new_img.SetDirection(img.GetDirection())
    return new_img


def get_distance(a_mask, b_mask):
    """Get the nearest distance between two masks.

    Args:
        a_mask: The first mask
        b_mask: The second mask
        dest: Working directory to output intermediate files

    Returns:
        None
    """

    # load lung mask from previous step
    try:
        a_mask = sitk.ReadImage(a_mask)
    except:
        print("File read failed " + a_mask)
        raise

    try:
        b_mask = sitk.ReadImage(b_mask)
    except:
        print("File read failed " + b_mask)
        raise

    # 1. Generate distance map from surface of Contour1
    smdm = sitk.SignedMaurerDistanceMapImageFilter()
    smdm.UseImageSpacingOn()
    smdm.InsideIsPositiveOff()
    smdm.SquaredDistanceOff()
    result = smdm.Execute(a_mask)

    # Subtract 1 from the mask, making 0's -> -1 and 1's -> 0.
    b_mask = b_mask - 1

    # Multiply this result by -10000, making -1's -> 10000.
    # It is assumed that 10000mm > than the max distance
    b_mask = b_mask * -10000

    result = result + sitk.Cast(b_mask, result.GetPixelIDValue())

    sif = sitk.StatisticsImageFilter()
    sif.Execute(result)
    dist = sif.GetMinimum()

    return dist


def generate_lung_mask(img, dest):
    """Generate initial airway mask (includes lungs).

    Args:
        image: path for CT image and mask destination.  The awful output filenames match
        the output from Antoine's C++ Masters code.

    Returns:
        None
    """

    print("Generating Lung Mask...")

    mid_img = sitk.GetArrayFromImage(img).astype(float)
    centre = ndimage.measurements.center_of_mass(mid_img)

    # get air mask external to body
    connected_threshold_filter = sitk.ConnectedThresholdImageFilter()
    connected_threshold_filter.SetSeed([0, 0, 0])
    connected_threshold_filter.SetLower(-10000)
    connected_threshold_filter.SetUpper(-300)
    result = connected_threshold_filter.Execute(img)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(dest + "/imageB.nii.gz")
    writer.Execute(result)

    # get body mask (remove couch) Seed this from centre of image.
    confidence_connected_image_filter = sitk.ConfidenceConnectedImageFilter()
    confidence_connected_image_filter.SetSeed([int(centre[1]), int(centre[2]), int(centre[0])])
    result_f = confidence_connected_image_filter.Execute(result)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(dest + "/imageF.nii.gz")
    writer.Execute(result_f)

    # get non-air voxels fom within body
    connected_threshold_filter = sitk.ConnectedThresholdImageFilter()
    connected_threshold_filter.SetSeed([251, 240, 65])
    connected_threshold_filter.SetLower(-300)
    connected_threshold_filter.SetUpper(10000)
    result = connected_threshold_filter.Execute(img)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(dest + "/imageG.nii.gz")
    writer.Execute(result)

    # problem might be intestinal air below lungs - need to remove this?
    invert_filter = sitk.InvertIntensityImageFilter()
    result_h = invert_filter.Execute(result)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(dest + "/imageH.nii.gz")
    writer.Execute(result_h)

    # not sure this is necessary
    binary_dilate = sitk.BinaryDilateImageFilter()
    binary_dilate.SetKernelRadius(3)
    result_j = binary_dilate.Execute(result_h)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(dest + "/imageJ.nii.gz")
    writer.Execute(result_j)

    # image j is ok (might need dilation)
    # image f is fine as well (just a body contour)
    add_image_filter = sitk.AddImageFilter()
    result_final = add_image_filter.Execute(result_j, result_f)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(dest + "/imageK.nii.gz")
    writer.Execute(result_final)

    result_final = invert_filter.Execute(result_final)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(dest + "/imageK2H.nii.gz")
    writer.Execute(result_final)

    # Just change 255 to 1 for binary label
    binary_threshold_filter = sitk.BinaryThresholdImageFilter()
    binary_threshold_filter.SetLowerThreshold(255)
    binary_threshold_filter.SetUpperThreshold(255)
    result_final = binary_threshold_filter.Execute(result_final)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(dest + "/imageFinal.nii.gz")
    writer.Execute(result_final)

    print("Generating Lung Mask... Done")

    return result_final


def generate_airway_mask(dest, img, lung_mask):
    """Generate final bronchus segmentation .

    Args:
        image: path for CT image and mask destination

    Returns:
        None
    """

    lung_mask_hu_values = [-750, -775, -800, -825, -850, -900, -700, -950, -650]
    distance_from_supu_slice_values = [3, 10, 20]
    expected_physical_size_range = [22000, 75000]

    z_size = img.GetDepth()

    # Identify airway start on superior slice
    label_shape = sitk.LabelIntensityStatisticsImageFilter()
    connected_component = sitk.ConnectedComponentImageFilter()
    connected_threshold_filter = sitk.ConnectedThresholdImageFilter()

    print("-------------------------------")

    label_shape.Execute(lung_mask, img)
    for label in label_shape.GetLabels():
        print(
            "summary of lung intensities.  Mean: "
            + str(label_shape.GetMean(label))
            + " sd: "
            + str(label_shape.GetStandardDeviation(label))
            + " median: "
            + str(label_shape.GetMedian(label))
        )

    loop_count = 0
    processed_correctly = False

    best_result = None
    best_result_sim = 100000000
    best_lung_mask_hu = 0

    for k in range(2):

        if processed_correctly and FAST_MODE:
            break

        if k == 1:
            # Try smoothing the lung mask - effects all tests below
            median_filter = sitk.MedianImageFilter()
            median_filter.SetRadius(1)
            lung_mask = median_filter.Execute(lung_mask)

        for distance_from_sup_slice in distance_from_supu_slice_values:

            if processed_correctly and FAST_MODE:
                break

            label_slice = lung_mask[
                :, :, z_size - distance_from_sup_slice - 10 : z_size - distance_from_sup_slice
            ]  # works for both cases 22 and 17
            img_slice = img[
                :, :, z_size - distance_from_sup_slice - 10 : z_size - distance_from_sup_slice
            ]

            connected = connected_component.Execute(label_slice)
            nda_connected = sitk.GetArrayFromImage(connected)
            # In case there are multiple air regions at the top, select the region with
            # the max elongation as the airway seed
            # Also check that the region has a minimum physical size, to filter out other
            # unexpected regions
            max_elong = 0
            airway_open = [0, 0, 0]
            label_shape.Execute(connected, img_slice)
            for label in label_shape.GetLabels():
                if (
                    label_shape.GetElongation(label) > max_elong
                    and label_shape.GetPhysicalSize(label) > 2000
                ):
                    centre = img.TransformPhysicalPointToIndex(label_shape.GetCentroid(label))
                    max_elong = label_shape.GetElongation(label)
                    airway_open = [int(centre[0]), int(centre[1]), int(centre[2])]

            # just check the opening is at the right location
            centroid_mask_val = lung_mask.GetPixel(airway_open[0], airway_open[1], airway_open[2])

            if centroid_mask_val == 0:
                print(
                    """Error locating trachea centroid.  Usually because of additional air features
                    on this slice"""
                )

                # No point in doing the airway segmentation here as airway opening wasn't detected
                continue

            print("*Airway opening: " + str(airway_open))
            print(
                "*Voxel HU at opening: "
                + str(lung_mask.GetPixel(airway_open[0], airway_open[1], airway_open[2]))
            )

            for lung_mask_hu in lung_mask_hu_values:

                print("--------------------------------------------")
                print("Extracting airways.  Iteration: " + str(loop_count))
                print("*Lung Mask HU: " + str(lung_mask_hu))
                print("*Slices from sup for airway opening: " + str(distance_from_sup_slice))
                if k == 1:
                    print("*Mask median smoothing on")
                loop_count += 1

                connected_threshold_filter.SetSeed(airway_open)
                connected_threshold_filter.SetLower(-2000)
                connected_threshold_filter.SetUpper(lung_mask_hu)
                result = connected_threshold_filter.Execute(img)

                writer = sitk.ImageFileWriter()
                writer.SetFileName(dest + "/airwaysMask.nii.gz")
                writer.Execute(result)

                # process in 2D - check label elongation and size.
                corina_slice = 0
                for idx_slice in range(z_size):
                    label_slice = result[:, :, idx_slice]
                    img_slice = img[:, :, idx_slice]
                    connected = connected_component.Execute(label_slice)
                    nda_connected = sitk.GetArrayFromImage(connected)
                    num_regions = nda_connected.max()

                    # Make sure the airway has branched
                    if num_regions < 2:
                        continue

                    label_shape.Execute(label_slice, img_slice)
                    for label in label_shape.GetLabels():
                        if (
                            label_shape.GetElongation(label) > 5
                            and label_shape.GetPhysicalSize(label) > 30
                        ):
                            corina_slice = idx_slice

                # crop from corina_slice + 20 slices (~4cm)
                if corina_slice == 0:
                    print("Failed to located carina.  Adjusting parameters ")

                else:
                    print(
                        " Cropping from slice: " + str(corina_slice) + " + 20 slices and dilating"
                    )
                    result = fast_mask(result, corina_slice + EXTEND_FROM_CARINA, z_size)
                    result = sitk.Cast(result, lung_mask.GetPixelIDValue())

                    # Dilate and check if the output is in the expected range
                    binary_dilate = sitk.BinaryDilateImageFilter()
                    binary_dilate.SetKernelRadius(2)
                    result = binary_dilate.Execute(result)

                    # check size of label - if it's too large the lungs have been included..
                    label_shape.Execute(result, img)
                    for label in label_shape.GetLabels():
                        airway_mask_physical_size = int(label_shape.GetPhysicalSize(label))
                        roundness = float(label_shape.GetRoundness(label))
                        elongation = float(label_shape.GetElongation(label))

                    this_processed_correctly = False
                    if airway_mask_physical_size > expected_physical_size_range[1]:
                        print(
                            " Airway Mask size failed (> "
                            + str(expected_physical_size_range[1])
                            + "): "
                            + str(airway_mask_physical_size)
                        )
                    elif airway_mask_physical_size < expected_physical_size_range[0]:
                        print(
                            " Airway Mask size failed (< "
                            + str(expected_physical_size_range[0])
                            + "): "
                            + str(airway_mask_physical_size)
                        )
                    else:
                        print(" Airway Mask size passed: " + str(airway_mask_physical_size))
                        processed_correctly = True
                        this_processed_correctly = True

                    print(" Roundness: " + str(roundness))
                    print(" Elongation: " + str(elongation))
                    # target_size = (
                    #     expected_physical_size_range[1] + expected_physical_size_range[0]
                    # ) / 2
                    #size_sim = abs(airway_mask_physical_size - target_size)

                    if roundness < best_result_sim and this_processed_correctly:
                        best_result_sim = roundness
                        best_result = result
                        best_lung_mask_hu = lung_mask_hu

    if not processed_correctly:
        print(" Unable to process correctly!!!")

    print("Selected Lung Mask HU: " + str(best_lung_mask_hu))

    return best_result
