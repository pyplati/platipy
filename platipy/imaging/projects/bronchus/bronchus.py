# Copyright 2020 CSIRO, University of New South Wales, Ingham Institute

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

from platipy.imaging.utils.lung import detect_holes, get_lung_mask


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


def generate_lung_mask(img):
    """Generate initial airway mask (includes lungs).

    Args:
        img: The SimpleITK CT image to segment the lungs in

    Returns:
        lung_mask: The mask containing the lung segmentation
    """

    print("Generating Lung Mask...")

    (label_image, labels) = detect_holes(img)
    lung_mask = get_lung_mask(label_image, labels)

    print("Generating Lung Mask... Done")

    return lung_mask


default_settings = {
    "fast_mode": True,
    "extend_from_carina_mm": 40,
    "minimum_tree_half_physical_size": 1000,
    "lung_mask_hu_values": [-750, -775, -800, -825, -850, -900, -700, -950, -650],
    "distance_from_supu_slice_values": [3, 10, 20],
    "expected_physical_size_range": [22000, 150000],
}


def generate_airway_mask(dest, img, lung_mask, config_dict=None):
    """Generate final bronchus segmentation .

    Args:
        image: path for CT image and mask destination

    Returns:
        None
    """

    if not config_dict:
        config_dict = default_settings

    fast_mode = config_dict["fast_mode"]
    extend_from_carina_mm = config_dict["extend_from_carina_mm"]
    lung_mask_hu_values = config_dict["lung_mask_hu_values"]
    minimum_tree_half_physical_size = config_dict["minimum_tree_half_physical_size"]
    distance_from_supu_slice_values = config_dict["distance_from_supu_slice_values"]
    expected_physical_size_range = config_dict["expected_physical_size_range"]

    z_size = img.GetDepth()
    z_spacing = img.GetSpacing()[2]
    extend_from_carina = round(extend_from_carina_mm / z_spacing)

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
    best_result_sim = 0
    best_lung_mask_hu = 0
    best_distance_from_sup_slice = 0

    for k in range(2):

        if processed_correctly and fast_mode:
            break

        if k == 1:
            # Try smoothing the lung mask - effects all tests below
            median_filter = sitk.MedianImageFilter()
            median_filter.SetRadius(1)
            lung_mask = median_filter.Execute(lung_mask)

        for distance_from_sup_slice in distance_from_supu_slice_values:

            if processed_correctly and fast_mode:
                break

            label_slice = lung_mask[
                :,
                :,
                z_size - distance_from_sup_slice - 10 : z_size - distance_from_sup_slice,
            ]  # works for both cases 22 and 17
            img_slice = img[
                :,
                :,
                z_size - distance_from_sup_slice - 10 : z_size - distance_from_sup_slice,
            ]

            connected = connected_component.Execute(label_slice)
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

                connected_threshold_filter.SetSeedList([airway_open])
                connected_threshold_filter.SetLower(-2000)
                connected_threshold_filter.SetUpper(lung_mask_hu)
                result = connected_threshold_filter.Execute(img)

                writer = sitk.ImageFileWriter()
                writer.SetFileName(dest + "/airwaysMask.nii.gz")
                writer.Execute(result)

                # Dilate and check if the output is in the expected range
                binary_dilate = sitk.BinaryDilateImageFilter()
                binary_dilate.SetKernelRadius(2)
                result = binary_dilate.Execute(result)

                # check size of label - if it's too large the lungs have been included..
                result = sitk.Cast(result, lung_mask.GetPixelIDValue())
                label_shape.Execute(result, img)
                airway_mask_physical_size = -1
                for label in label_shape.GetLabels():
                    airway_mask_physical_size = int(label_shape.GetPhysicalSize(label))
                    roundness = float(label_shape.GetRoundness(label))
                    elongation = float(label_shape.GetElongation(label))

                this_processed_correctly = False
                if airway_mask_physical_size < 0:
                    print("No labels found in mask")
                    continue
                elif airway_mask_physical_size > expected_physical_size_range[1]:
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
                # size_sim = abs(airway_mask_physical_size - target_size)

                if airway_mask_physical_size > best_result_sim and this_processed_correctly:
                    best_result_sim = airway_mask_physical_size
                    best_result = result
                    best_lung_mask_hu = lung_mask_hu
                    best_distance_from_sup_slice = distance_from_sup_slice

    if not processed_correctly:
        print(" Unable to process correctly!!!")

    print("Selected Lung Mask HU: " + str(best_lung_mask_hu))

    # process in 2D - check label elongation and size.
    corina_slice = -1
    lssif = sitk.LabelShapeStatisticsImageFilter()
    for idx_slice in range(z_size - best_distance_from_sup_slice, 0, -1):

        cut_mask = fast_mask(best_result, idx_slice, z_size)
        cut_mask = sitk.Cast(cut_mask, lung_mask.GetPixelIDValue())

        label_image = connected_component.Execute(cut_mask)

        num_regions = connected_component.GetObjectCount()

        if num_regions == 2:
            lssif.Execute(label_image)

            phys_size_0 = int(lssif.GetPhysicalSize(1))
            phys_size_1 = int(lssif.GetPhysicalSize(2))

            if (
                phys_size_0 > minimum_tree_half_physical_size
                and phys_size_1 > minimum_tree_half_physical_size
            ):

                corina_slice = idx_slice
                break

    if corina_slice >= 0:
        print(f" Cropping from slice: {corina_slice} + {extend_from_carina} slices")
        best_result = fast_mask(best_result, corina_slice + extend_from_carina, z_size)

    best_result = sitk.Cast(best_result, lung_mask.GetPixelIDValue())

    return best_result
