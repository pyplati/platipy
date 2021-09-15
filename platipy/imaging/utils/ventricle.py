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

import copy

import numpy as np

import SimpleITK as sitk

from platipy.imaging.label.utils import get_com

from platipy.imaging.utils.geometry import vector_angle, rotate_image

from platipy.imaging.utils.crop import crop_to_roi, label_to_roi


def extract(template_img, angles, angle_min, angle_max, loc_x, loc_y, cw=False):
    """
    Utility function to extract relevant voxels from a mask based on polar coordinates
    """
    # Get template array
    template_arr = sitk.GetArrayViewFromImage(template_img)
    # Get the segment array
    segment_arr = np.zeros_like(template_arr)

    # Define the condition list
    if cw:
        in_segment_condition = (angles <= angle_min) | (angles >= angle_max)
    else:
        in_segment_condition = (angles <= angle_max) & (angles >= angle_min)

    # Extract matching voxels
    segment_arr[loc_y[in_segment_condition], loc_x[in_segment_condition]] = 1
    # Convert to image
    segment_img = sitk.GetImageFromArray(segment_arr)
    segment_img.CopyInformation(template_img)

    return segment_img


def generate_left_ventricle_segments(
    contours,
    label_left_ventricle="LEFTVENTRICLE",
    label_right_ventricle="RIGHTVENTRICLE",
    label_mitral_valve="MITRALVALVE",
    label_heart="WHOLEHEART",
    myocardium_thickness_mm=10,
    edge_correction_mm=0,
):
    """
    Generates the 17 segments of the left vetricle

    This functions works as follows:
        1.  Heart volume is rotated to align the long axis to the z Cartesian (physical) space.
            Usually means it aligns with the axial axis (for normal simulation CT)
        2.  Left ventricle is divided into thirds along the long axis
        3.  Myocardium is defined as the outer 10mm
        4.  Geometric operations are used to define the segments
        6.  Everything is rotated back to the normal orientation
        7.  Some post-processing *magic*

    Args:
        contours (dict): A dictionary containing strings (label names) as keys and SimpleITK.Image
            (masks) as values. Must contain at least the LV, RV, MV, and whole heart.
        label_left_ventricle (str): The name for the left ventricle mask (contour)
        label_right_ventricle (str): The name for the right ventricle mask (contour)
        label_mitral_valve (str): The name for the mitral valve mask (contour)
        label_heart (str): The name for the heart mask (contour)
        myocardium_thickness_mm (int, optional): Moycardial thickness, in millimetres.
            Defaults to 10.
        edge_correction_mm (float, optional): Can be used to give a bit of separation between
            segments. Defaults to 1.

    Returns:
        dict : The left ventricle segment dictionary, with labels (int) as keys and the binary
        label defining the segment (SimpleITK.Image) as values.
    """

    """
    Crop the images
    Rotate to the cardiac axis
    """
    working_contours = copy.deepcopy(contours)
    output_contours = {}

    # Some conversions
    erode_img = [
        int(myocardium_thickness_mm / i)
        for i in working_contours[label_left_ventricle].GetSpacing()
    ]

    # Crop to the smallest volume possible to make it FAST
    cb_size, cb_index = label_to_roi(
        working_contours[label_heart] > 0,
        expansion_mm=(30, 30, 60),  # Better to make it a bit bigger to be safe
    )

    for label in contours:
        working_contours[label] = crop_to_roi(contours[label], cb_size, cb_index)

    # Initially we should reorient based on the cardiac axis
    label_orient = (working_contours[label_heart]) > 0  # + working_contours[label_left_atrium]

    lsf = sitk.LabelShapeStatisticsImageFilter()  # this will be used throughout
    lsf.Execute(label_orient)
    cardiac_axis = np.array(lsf.GetPrincipalAxes(1)[:3])  # First principal axis approx. long axis

    # The principal axis isn't guaranteed to point from base to apex
    # If is points apex to base, we have to invert it
    # So check that here
    if cardiac_axis[2] < 0:
        cardiac_axis = -1 * cardiac_axis

    rotation_angle = vector_angle(cardiac_axis, (0, 0, 1))
    rotation_axis = np.cross(cardiac_axis, (0, 0, 1))
    rotation_centre = get_com(working_contours[label_heart], real_coords=True)

    for label in contours:
        working_contours[label] = rotate_image(
            working_contours[label],
            rotation_centre=rotation_centre,
            rotation_axis=rotation_axis,
            rotation_angle_radians=rotation_angle,
            interpolation=sitk.sitkNearestNeighbor,
            default_value=0,
        )

    """
    Compute the myocardium for the whole LV volume

    Divide this volume into thirds (from MV COM -> LV apex)

    Iterate through each slice
        1. Find the anterior insertion of the RV
        This defines theta_0
        2. Find the centre of the LV slice
        This defines r_0 (origin)
        3. Convert each myocardium label loc to polar coords
        4. Assign each label to the appropriate LV segments    
        
    """

    # First, let's just extract the myocardium
    label_lv_inner = sitk.BinaryErode(working_contours[label_left_ventricle], erode_img)
    label_lv_myo = working_contours[label_left_ventricle] - label_lv_inner

    # Mask the myo to a dilation of the blood pool
    # This helps improve shape consistency
    label_lv_myo_mask = sitk.BinaryDilate(label_lv_inner, erode_img)
    label_lv_myo = sitk.Mask(label_lv_myo, label_lv_myo_mask)

    # Computing limits for division into thirds
    # [xstart, ystart, zstart, xsize, ysize, zsize]
    # For the limits, we will use the centre of mass of the MV to the LV apex
    # The inner limit is used to assign the top portion (basal) of the LV to the anterior segment
    lsf.Execute(label_lv_inner)
    _, _, inf_limit_lv, _, _, _ = lsf.GetRegion(1)

    lsf.Execute(label_lv_myo)
    _, _, inf_limit_lv_myo, _, _, extent_lv_myo = lsf.GetRegion(1)
    sup_limit_lv_myo = inf_limit_lv_myo + extent_lv_myo

    com_mv, _, _ = get_com(working_contours[label_mitral_valve])

    extent = sup_limit_lv_myo - inf_limit_lv  # com_mv - inf_limit_lv
    dc = int(extent / 3)

    # Define limits (cut LV into thirds)
    apical_extent = inf_limit_lv + dc
    mid_extent = inf_limit_lv + 2 * dc
    basal_extent = inf_limit_lv + 3 * dc

    # Segment 17
    label_lv_myo_apex = label_lv_myo * 1  # make a copy
    label_lv_myo_apex[:, :, inf_limit_lv:] = 0

    # The apical segment
    label_lv_myo_apical = label_lv_myo * 1  # make a copy
    label_lv_myo_apical[:, :, :inf_limit_lv] = 0
    label_lv_myo_apical[:, :, apical_extent:] = 0

    # The mid segment
    label_lv_myo_mid = label_lv_myo * 1  # make a copy
    label_lv_myo_mid[:, :, :apical_extent] = 0
    label_lv_myo_mid[:, :, mid_extent:] = 0

    # The basal segment
    label_lv_myo_basal = label_lv_myo * 1  # make a copy
    label_lv_myo_basal[:, :, :mid_extent] = 0
    label_lv_myo_basal[:, :, basal_extent:] = 0

    """
    Now we generate the segments
    """

    for i in range(17):
        working_contours[i + 1] = 0 * working_contours[label_heart]

    working_contours[17] = label_lv_myo_apex

    # We are now going to compute the segments in cylindical sections
    # First up - apical slices
    for n in range(inf_limit_lv, apical_extent):

        label_lv_myo_slice = label_lv_myo[:, :, n]

        # We will need numpy arrays here
        arr_lv_myo_slice = sitk.GetArrayViewFromImage(label_lv_myo_slice)
        loc_y, loc_x = np.where(arr_lv_myo_slice)

        # Now the origin
        y_0, x_0 = get_com(label_lv_myo_slice)

        # Compute the angle(s)
        theta = -1.0 * np.arctan2(loc_y - y_0, loc_x - x_0)
        # Convert to [0,2*np.pi]
        theta[theta < 0] += 2 * np.pi

        # Now assign to different segments
        working_contours[13][:, :, n] = extract(
            label_lv_myo_slice, theta, 1 * np.pi / 4, 3 * np.pi / 4, loc_x, loc_y
        )
        working_contours[14][:, :, n] = extract(
            label_lv_myo_slice, theta, 3 * np.pi / 4, 5 * np.pi / 4, loc_x, loc_y
        )
        working_contours[15][:, :, n] = extract(
            label_lv_myo_slice, theta, 5 * np.pi / 4, 7 * np.pi / 4, loc_x, loc_y
        )
        working_contours[16][:, :, n] = extract(
            label_lv_myo_slice, theta, 1 * np.pi / 4, 7 * np.pi / 4, loc_x, loc_y, cw=True
        )

    # Second up - mid slices
    for n in range(apical_extent, mid_extent):

        label_rv_slice = working_contours[label_right_ventricle][:, :, n]
        label_lv_myo_slice = label_lv_myo[:, :, n]

        # We will need numpy arrays here
        arr_lv_myo_slice = sitk.GetArrayViewFromImage(label_lv_myo_slice)
        loc_y, loc_x = np.where(arr_lv_myo_slice)

        # Now the origin
        y_0, x_0 = get_com(label_lv_myo_slice)

        # We need the angle for the anterior RV insertion
        # First, locate the tip
        _, loc_rv_y, loc_rv_x = np.where(
            sitk.GetArrayViewFromImage(working_contours[label_right_ventricle])
        )
        loc_rv_tip_x = loc_rv_x.max()
        loc_rv_tip_y = np.median(loc_rv_y[np.where(loc_rv_x == loc_rv_tip_x)]).astype(int)

        # Second, compute the angle
        theta_0 = np.arctan2(loc_rv_tip_y - y_0, loc_rv_tip_x - x_0)

        # Compute the angle(s)
        theta = theta_0 - np.arctan2(loc_y - y_0, loc_x - x_0)
        # Convert to [0,2*np.pi]
        theta[theta < 0] += 2 * np.pi

        # Now assign to different segments
        working_contours[8][:, :, n] = extract(
            label_lv_myo_slice, theta, 0, np.pi / 3, loc_x, loc_y
        )
        working_contours[9][:, :, n] = extract(
            label_lv_myo_slice, theta, 1 * np.pi / 3, 2 * np.pi / 3, loc_x, loc_y
        )
        working_contours[10][:, :, n] = extract(
            label_lv_myo_slice, theta, 2 * np.pi / 3, 3 * np.pi / 3, loc_x, loc_y
        )
        working_contours[11][:, :, n] = extract(
            label_lv_myo_slice, theta, 3 * np.pi / 3, 4 * np.pi / 3, loc_x, loc_y
        )
        working_contours[12][:, :, n] = extract(
            label_lv_myo_slice, theta, 4 * np.pi / 3, 5 * np.pi / 3, loc_x, loc_y
        )
        working_contours[7][:, :, n] = extract(
            label_lv_myo_slice, theta, 5 * np.pi / 3, 2 * np.pi, loc_x, loc_y
        )

    # Third up - basal slices
    for n in range(mid_extent, basal_extent):

        label_rv_slice = working_contours[label_right_ventricle][:, :, n]
        label_lv_myo_slice = label_lv_myo[:, :, n]

        # We will need numpy arrays here
        arr_lv_myo_slice = sitk.GetArrayViewFromImage(label_lv_myo_slice)
        loc_y, loc_x = np.where(arr_lv_myo_slice)

        # Now the origin
        y_0, x_0 = get_com(label_lv_myo_slice)

        # We need the angle for the anterior RV insertion
        # First, locate the tip
        if sitk.GetArrayViewFromImage(label_rv_slice).sum() == 0:
            # Use the last slice value
            # i.e. do nothing this iteration
            pass
        else:
            _, loc_rv_y, loc_rv_x = np.where(
                sitk.GetArrayViewFromImage(working_contours[label_right_ventricle])
            )
            loc_rv_tip_x = loc_rv_x.max()
            loc_rv_tip_y = np.median(loc_rv_y[np.where(loc_rv_x == loc_rv_tip_x)]).astype(int)

        # Second, compute the angle
        theta_0 = np.arctan2(loc_rv_tip_y - y_0, loc_rv_tip_x - x_0)

        # Compute the angle(s)
        theta = theta_0 - np.arctan2(loc_y - y_0, loc_x - x_0)
        # Convert to [0,2*np.pi]
        theta[theta < 0] += 2 * np.pi

        # Now assign to different segments
        working_contours[2][:, :, n] = extract(
            label_lv_myo_slice, theta, 0, np.pi / 3, loc_x, loc_y
        )
        working_contours[3][:, :, n] = extract(
            label_lv_myo_slice, theta, 1 * np.pi / 3, 2 * np.pi / 3, loc_x, loc_y
        )
        working_contours[4][:, :, n] = extract(
            label_lv_myo_slice, theta, 2 * np.pi / 3, 3 * np.pi / 3, loc_x, loc_y
        )
        working_contours[5][:, :, n] = extract(
            label_lv_myo_slice, theta, 3 * np.pi / 3, 4 * np.pi / 3, loc_x, loc_y
        )
        working_contours[6][:, :, n] = extract(
            label_lv_myo_slice, theta, 4 * np.pi / 3, 5 * np.pi / 3, loc_x, loc_y
        )
        working_contours[1][:, :, n] = extract(
            label_lv_myo_slice, theta, 5 * np.pi / 3, 2 * np.pi, loc_x, loc_y
        )

    # Rotate back to the original reference space
    for segment in range(17):
        new_structure = rotate_image(
            working_contours[segment + 1],
            rotation_centre=rotation_centre,
            rotation_axis=rotation_axis,
            rotation_angle_radians=-1.0 * rotation_angle,
            interpolation=sitk.sitkNearestNeighbor,
            default_value=0,
        )

        #     new_structure = sitk.BinaryMorphologicalClosing(new_structure, hole_fill_img)
        #     new_structure = sitk.RelabelComponent(sitk.ConnectedComponent(new_structure)) == 1
        if edge_correction_mm > 0:
            new_structure = sitk.BinaryErode(
                new_structure,
                [int(edge_correction_mm / j) for j in new_structure.GetSpacing()[:2]]
                + [
                    0,
                ],
            )
        new_structure = sitk.Paste(
            contours[label_heart],
            new_structure,
            new_structure.GetSize(),
            (0, 0, 0),
            cb_index,
        )

        output_contours[segment + 1] = new_structure

    return output_contours
