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

from platipy.imaging.utils.crop import crop_to_roi, label_to_roi

from platipy.imaging.utils.geometry import vector_angle


def extract(
    template_img,
    angles,
    radii,
    angle_min,
    angle_max,
    loc_x,
    loc_y,
    cw=False,
    radius_min=0,
    min_area_mm2=25,
):
    """
    Utility function to extract relevant voxels from a mask based on polar coordinates
    """
    # Get template array
    template_arr = sitk.GetArrayViewFromImage(template_img)
    # Get the segment array
    segment_arr = np.zeros_like(template_arr)

    # Define the condition list
    radius_min = radius_min

    if cw:
        in_segment_condition = (angles <= angle_min) | (angles >= angle_max)
        in_segment_condition &= radii >= radius_min
    else:
        in_segment_condition = (angles <= angle_max) & (angles >= angle_min)
        in_segment_condition &= radii >= radius_min

    # Extract matching voxels
    segment_arr[loc_y[in_segment_condition], loc_x[in_segment_condition]] = 1

    # Convert to image
    segment_img = sitk.GetImageFromArray(segment_arr)
    segment_img.CopyInformation(template_img)

    # Make sure area exceeds lower bound
    area = segment_arr.sum() * np.product(segment_img.GetSpacing())
    if area < min_area_mm2:
        segment_img *= 0

    return segment_img


def generate_left_ventricle_segments(
    contours,
    label_left_ventricle="LEFTVENTRICLE",
    label_left_atrium="LEFTATRIUM",
    label_right_ventricle="RIGHTVENTRICLE",
    label_mitral_valve="MITRALVALVE",
    label_heart="WHOLEHEART",
    myocardium_thickness_mm=10,
    hole_fill_mm=3,
    optimiser_tol_degrees=1,
    optimiser_max_iter=10,
    min_area_mm2=50,
    verbose=False,
):
    """
    Generates the 17 segments of the left vetricle

    This functions works as follows:
        1.  Heart volume is rotated to align the long axis to the z Cartesian (physical) space.
            Usually means it aligns with the axial axis (for normal simulation CT)
        2.  An optimiser adjusts the orientation to refine this alignment to the vector defined by
            MV COM - LV apex axis (long axis)
        3.  Left ventricle is divided into thirds along the long axis
        4.  Myocardium is defined as the outer 10mm
        5.  Geometric operations are used to define the segments
        6.  Everything is rotated back to the normal orientation
        7.  Some post-processing *magic*

    Args:
        contours (dict): A dictionary containing strings (label names) as keys and SimpleITK.Image
            (masks) as values. Must contain at least the LV, RV, MV, and whole heart.
        label_left_ventricle (str): The name for the left ventricle mask (contour)
        label_left_atrium (str): The name for the left atrium mask (contour)
        label_right_ventricle (str): The name for the right ventricle mask (contour)
        label_mitral_valve (str): The name for the mitral valve mask (contour)
        label_heart (str): The name for the heart mask (contour)
        myocardium_thickness_mm (float, optional): Moycardial thickness, in millimetres.
            Defaults to 10.
        hole_fill_mm (float, optional): Holes smaller than this get filled in. Defaults to 3.
        optimiser_tol_degrees (float, optional): Optimiser tolerance (change in angle per iter).
            Defaults to 1, which typically requires 3-4 iterations.
        optimiser_max_iter (int, optional): Maximum optimiser iterations. Defaults to 10
        verbose (bool, optional): Print of information for debugging. Defaults to False.

    Returns:
        dict : The left ventricle segment dictionary, with labels (int) as keys and the binary
        label defining the segment (SimpleITK.Image) as values.
    """

    if verbose:
        print("Beginning LV segmentation algorithm.")

    # Initial set up
    label_list = [
        label_left_ventricle,
        label_left_atrium,
        label_right_ventricle,
        label_mitral_valve,
        label_heart,
    ]
    working_contours = copy.deepcopy({s: contours[s] for s in label_list})
    output_contours = {}
    overall_transform_list = []

    # Some conversions
    erode_img = [
        int(myocardium_thickness_mm / i)
        for i in working_contours[label_left_ventricle].GetSpacing()
    ]
    hole_fill_img = [int(hole_fill_mm / i) for i in working_contours[label_heart].GetSpacing()]

    """
    Module 1 - Preparation
    Crop the images
    Rotate to the cardiac axis
    """
    # Crop to the smallest volume possible to make it FAST
    cb_size, cb_index = label_to_roi(
        working_contours[label_heart] > 0,
        expansion_mm=(30, 30, 60),  # Better to make it a bit bigger to be safe
    )

    for label in contours:
        working_contours[label] = crop_to_roi(contours[label], cb_size, cb_index)

    if verbose:
        print("Module 1: Cropping and initial alignment.")
        vol_before = np.product(contours[label_heart].GetSpacing())
        vol_after = np.product(working_contours[label_heart].GetSpacing())
        print(f"  Images cropped. Volume reduction: {vol_before/vol_after:.3f}")

    # Initially we should reorient based on the cardiac axis
    label_orient = (
        working_contours[label_left_ventricle] + working_contours[label_left_atrium]
    ) > 0

    lsf = sitk.LabelShapeStatisticsImageFilter()  # this will be used throughout
    lsf.Execute(label_orient)
    cardiac_axis = np.array(lsf.GetPrincipalAxes(1)[:3])  # First principal axis approx. long axis

    # The principal axis isn't guaranteed to point from base to apex
    # If is points apex to base, we have to invert it
    # So check that here
    if cardiac_axis[2] < 0:
        cardiac_axis = -1 * cardiac_axis

    rotation_angle = vector_angle(cardiac_axis[::-1], (0, 0, 1))
    rotation_axis = np.cross(cardiac_axis[::-1], (0, 0, 1))
    rotation_centre = get_com(label_orient, real_coords=True)

    if verbose:
        print("  Alignment computed.")
        print("    Cardiac axis:    ", cardiac_axis)
        print("    Rotation axis:   ", rotation_axis)
        print("    Rotation angle:  ", rotation_angle)
        print("    Rotation centre: ", rotation_centre)

    rotation_transform = sitk.VersorRigid3DTransform()
    rotation_transform.SetCenter(rotation_centre)
    rotation_transform.SetRotation(rotation_axis, rotation_angle)

    overall_transform_list.append(rotation_transform)

    for label in contours:
        working_contours[label] = sitk.Resample(
            working_contours[label],
            rotation_transform,
            sitk.sitkNearestNeighbor,
            0,
            working_contours[label].GetPixelID(),
        )

    """
    Module 2 - LV orientation alignment
    We use a very simple optimisation regime to enable robust computation of the LV apex
    We compute the vector from the MV COM to the LV apex
    This will be used for orientation (i.e. the long axis)
    """
    optimiser_tol_radians = optimiser_tol_degrees * np.pi / 180

    n = 0

    if verbose:
        print("Module 2: LV orientation alignment.")
        print("  Optimiser tolerance (degrees) =", optimiser_tol_degrees)
        print("  Beginning alignment process")

    while n < optimiser_max_iter and np.abs(rotation_angle) > optimiser_tol_radians:

        n += 1

        # Find the LV apex
        lv_locations = np.where(sitk.GetArrayViewFromImage(working_contours[label_left_ventricle]))
        lv_apex_z = lv_locations[0].min()
        lv_apex_y = lv_locations[1][lv_locations[0] == lv_apex_z].mean()
        lv_apex_x = lv_locations[2][lv_locations[0] == lv_apex_z].mean()
        lv_apex_loc = np.array([lv_apex_x, lv_apex_y, lv_apex_z])

        # Get the MV COM
        mv_com = np.array(get_com(working_contours[label_mitral_valve], real_coords=True))

        # Define the LV axis
        lv_apex_loc_img = np.array(
            working_contours[label_left_ventricle].TransformContinuousIndexToPhysicalPoint(
                lv_apex_loc.tolist()
            )
        )
        lv_axis = lv_apex_loc_img - mv_com

        # Compute the rotation parameters
        rotation_axis = np.cross(lv_axis, (0, 0, 1))
        rotation_angle = vector_angle(lv_axis, (0, 0, 1))
        rotation_centre = 0.5 * (
            mv_com + lv_apex_loc_img
        )  # get_com(working_contours[label_left_ventricle], real_coords=True)

        rotation_transform = sitk.VersorRigid3DTransform()
        rotation_transform.SetCenter(rotation_centre)
        rotation_transform.SetRotation(rotation_axis, rotation_angle)

        overall_transform_list.append(rotation_transform)

        if verbose:
            print("    N:               ", n)
            print("    LV apex:         ", lv_apex_loc_img)
            print("    MV COM:          ", mv_com)
            print("    LV axis:         ", lv_axis)
            print("    Rotation axis:   ", rotation_axis)
            print("    Rotation centre: ", rotation_centre)
            print("    Rotation angle:  ", rotation_angle)

        for label in contours:
            working_contours[label] = sitk.Resample(
                working_contours[label],
                rotation_transform,
                sitk.sitkNearestNeighbor,
                0,
                working_contours[label].GetPixelID(),
            )

    """
    Module 3 - Compute the myocardium for the whole LV volume

    Divide this volume into thirds (from MV COM -> LV apex)        
    """

    if verbose:
        print("Module 3: Myocardium generation.")

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
    _, _, inf_limit_lv, _, _, extent = lsf.GetRegion(1)

    com_mv, _, _ = get_com(working_contours[label_mitral_valve])

    extent = com_mv - inf_limit_lv
    dc = int(extent / 3)

    # Define limits (cut LV into thirds)
    apical_extent = inf_limit_lv + dc
    mid_extent = inf_limit_lv + 2 * dc
    basal_extent = com_mv  # more complete coverage

    if verbose:
        print("  Apex (long axis) slice:      ", inf_limit_lv)
        print("  Apical section extent slice: ", apical_extent)
        print("  Mid section extent slice:    ", mid_extent)
        print("  Basal section extent slice:  ", basal_extent)
        print("    DeltaCut (DC): ", dc)
        print("    Extent:        ", extent)

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
    Module 4 - Generate 17 segments

        1. Find the basal (anterior) insertion of the RV
            This defines theta_0
        2. Find the baseline angle for the apical section
            This defines thera_0_apical
        3. Iterate though each section (apical, mid, basal):
            a. Convert each myocardium label loc to polar coords
            b. Assign each label to the appropriate LV segments
    """

    if verbose:
        print("Module 4: Segment generation.")

    # We need the angle for the basal RV insertion
    # This is the most counter-clockwise RV location
    # First, retrieve the most basal 5 slices
    loc_rv_z, loc_rv_y, loc_rv_x = np.where(
        sitk.GetArrayViewFromImage(working_contours[label_right_ventricle])
    )
    loc_rv_z_basal = np.arange(mid_extent, mid_extent + 5)

    if verbose:
        print("  RV basal slices: ", loc_rv_z_basal)

    theta_rv_insertion = []
    for z in loc_rv_z_basal:
        # Now get all the x and y positions
        loc_rv_basal_x = loc_rv_x[np.where(np.in1d(loc_rv_z, z))]
        loc_rv_basal_y = loc_rv_y[np.where(np.in1d(loc_rv_z, z))]

        # Now define the LV COM on each slice
        lv_com = get_com(working_contours[label_left_ventricle][:, :, int(z)])
        lv_com_basal_x = lv_com[1]
        lv_com_basal_y = lv_com[0]

        # Compute the angle
        theta_rv = np.arctan2(lv_com_basal_y - loc_rv_basal_y, loc_rv_basal_x - lv_com_basal_x)
        theta_rv[theta_rv < 0] += 2 * np.pi
        theta_rv_insertion.append(theta_rv.min())

    theta_0 = np.median(theta_rv_insertion)

    if verbose:
        print("  RV insertion angle (basal section): ", theta_0)

    # We also need the angle in the apical section for accurate segmentation
    lv_com_apical_list = []
    rv_com_apical_list = []
    for n in range(inf_limit_lv, apical_extent):
        lv_com_apical_list.append(get_com(working_contours[label_left_ventricle][:, :, n]))
        rv_com_apical_list.append(get_com(working_contours[label_right_ventricle][:, :, n]))

    lv_com_apical = np.mean(lv_com_apical_list, axis=0)
    rv_com_apical = np.mean(rv_com_apical_list, axis=0)

    theta_0_apical = np.arctan2(
        lv_com_apical[0] - rv_com_apical[0], rv_com_apical[1] - lv_com_apical[1]
    )

    if verbose:
        print(" Apical LV-RV COM angle: ", theta_0_apical)

    for i in range(17):
        working_contours[i + 1] = 0 * working_contours[label_heart]

    working_contours[17] = label_lv_myo_apex

    if verbose:
        print("  Computing apical segments")
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
        theta = -np.arctan2(loc_y - y_0, loc_x - x_0) - theta_0_apical
        # Convert to [0,2*np.pi]
        theta[theta < 0] += 2 * np.pi

        # Compute the radii
        radii = np.sqrt((loc_y - y_0) ** 2 + (loc_x - x_0) ** 2)

        # Now assign to different segments
        working_contours[13][:, :, n] = extract(
            label_lv_myo_slice,
            theta,
            radii,
            5 * np.pi / 4,
            7 * np.pi / 4,
            loc_x,
            loc_y,
            min_area_mm2=min_area_mm2,
        )
        working_contours[14][:, :, n] = extract(
            label_lv_myo_slice,
            theta,
            radii,
            1 * np.pi / 4,
            7 * np.pi / 4,
            loc_x,
            loc_y,
            cw=True,
            min_area_mm2=min_area_mm2,
        )
        working_contours[15][:, :, n] = extract(
            label_lv_myo_slice,
            theta,
            radii,
            1 * np.pi / 4,
            3 * np.pi / 4,
            loc_x,
            loc_y,
            min_area_mm2=min_area_mm2,
        )
        working_contours[16][:, :, n] = extract(
            label_lv_myo_slice,
            theta,
            radii,
            3 * np.pi / 4,
            5 * np.pi / 4,
            loc_x,
            loc_y,
            min_area_mm2=min_area_mm2,
        )

    if verbose:
        print("  Computing mid segments")
    # Second up - mid slices
    for n in range(apical_extent, mid_extent):

        label_lv_myo_slice = label_lv_myo[:, :, n]

        # We will need numpy arrays here
        arr_lv_myo_slice = sitk.GetArrayViewFromImage(label_lv_myo_slice)
        loc_y, loc_x = np.where(arr_lv_myo_slice)

        # Now the origin
        y_0, x_0 = get_com(label_lv_myo_slice)

        # Compute the angle(s)
        theta = -np.arctan2(loc_y - y_0, loc_x - x_0) - theta_0
        # Convert to [0,2*np.pi]
        theta[theta < 0] += 2 * np.pi

        # Compute the radii
        radii = np.sqrt((loc_y - y_0) ** 2 + (loc_x - x_0) ** 2)

        # Now assign to different segments
        working_contours[8][:, :, n] = extract(
            label_lv_myo_slice, theta, radii, 0, np.pi / 3, loc_x, loc_y, min_area_mm2=min_area_mm2
        )
        working_contours[9][:, :, n] = extract(
            label_lv_myo_slice,
            theta,
            radii,
            1 * np.pi / 3,
            2 * np.pi / 3,
            loc_x,
            loc_y,
            min_area_mm2=min_area_mm2,
        )
        working_contours[10][:, :, n] = extract(
            label_lv_myo_slice,
            theta,
            radii,
            2 * np.pi / 3,
            3 * np.pi / 3,
            loc_x,
            loc_y,
            min_area_mm2=min_area_mm2,
        )
        working_contours[11][:, :, n] = extract(
            label_lv_myo_slice,
            theta,
            radii,
            3 * np.pi / 3,
            4 * np.pi / 3,
            loc_x,
            loc_y,
            min_area_mm2=min_area_mm2,
        )
        working_contours[12][:, :, n] = extract(
            label_lv_myo_slice,
            theta,
            radii,
            4 * np.pi / 3,
            5 * np.pi / 3,
            loc_x,
            loc_y,
            min_area_mm2=min_area_mm2,
        )
        working_contours[7][:, :, n] = extract(
            label_lv_myo_slice,
            theta,
            radii,
            5 * np.pi / 3,
            2 * np.pi,
            loc_x,
            loc_y,
            min_area_mm2=min_area_mm2,
        )

    if verbose:
        print("  Computing basal segments")
    # Third up - basal slices
    for n in range(mid_extent, basal_extent):

        label_lv_myo_slice = label_lv_myo[:, :, n]

        # We will need numpy arrays here
        arr_lv_myo_slice = sitk.GetArrayViewFromImage(label_lv_myo_slice)
        loc_y, loc_x = np.where(arr_lv_myo_slice)

        # Now the origin
        y_0, x_0 = get_com(label_lv_myo_slice)

        # Compute the angle(s)
        theta = -np.arctan2(loc_y - y_0, loc_x - x_0) - theta_0
        # Convert to [0,2*np.pi]
        theta[theta < 0] += 2 * np.pi

        # Compute the radii
        radii = np.sqrt((loc_y - y_0) ** 2 + (loc_x - x_0) ** 2)

        # Now assign to different segments
        working_contours[2][:, :, n] = extract(
            label_lv_myo_slice,
            theta,
            radii,
            0,
            np.pi / 3,
            loc_x,
            loc_y,
            radius_min=15,
            min_area_mm2=min_area_mm2,
        )
        working_contours[3][:, :, n] = extract(
            label_lv_myo_slice,
            theta,
            radii,
            1 * np.pi / 3,
            2 * np.pi / 3,
            loc_x,
            loc_y,
            radius_min=15,
            min_area_mm2=min_area_mm2,
        )
        working_contours[4][:, :, n] = extract(
            label_lv_myo_slice,
            theta,
            radii,
            2 * np.pi / 3,
            3 * np.pi / 3,
            loc_x,
            loc_y,
            radius_min=15,
            min_area_mm2=min_area_mm2,
        )
        working_contours[5][:, :, n] = extract(
            label_lv_myo_slice,
            theta,
            radii,
            3 * np.pi / 3,
            4 * np.pi / 3,
            loc_x,
            loc_y,
            radius_min=15,
            min_area_mm2=min_area_mm2,
        )
        working_contours[6][:, :, n] = extract(
            label_lv_myo_slice,
            theta,
            radii,
            4 * np.pi / 3,
            5 * np.pi / 3,
            loc_x,
            loc_y,
            radius_min=15,
            min_area_mm2=min_area_mm2,
        )
        working_contours[1][:, :, n] = extract(
            label_lv_myo_slice,
            theta,
            radii,
            5 * np.pi / 3,
            2 * np.pi,
            loc_x,
            loc_y,
            radius_min=15,
            min_area_mm2=min_area_mm2,
        )

    """
    Module 5 - re-orientation into image space

    We perform the total inverse transformation, and paste the labels back into the image space
    """

    if verbose:
        print("  Module 5: Re-orientation.")

    # Compute the total transform
    overall_transform = sitk.CompositeTransform(overall_transform_list)
    inverse_transform = overall_transform.GetInverse()

    # Rotate back to the original reference space
    for segment in range(17):
        new_structure = sitk.Resample(
            working_contours[segment + 1],
            inverse_transform,
            sitk.sitkNearestNeighbor,
            0,
        )

        if hole_fill_mm > 0:
            new_structure = sitk.BinaryMorphologicalClosing(new_structure, hole_fill_img)

        new_structure = sitk.Paste(
            contours[label_heart] * 0,
            new_structure,
            new_structure.GetSize(),
            (0, 0, 0),
            cb_index,
        )

        output_contours[segment + 1] = new_structure

    if verbose:
        print("Complete!")

    return output_contours
