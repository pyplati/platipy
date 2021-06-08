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

import numpy as np

import SimpleITK as sitk

from platipy.imaging.label.utils import get_com

from platipy.imaging.generation.image import insert_cylinder_image

from platipy.imaging.utils.geometry import vector_angle, rotate_image


def generate_valve_from_great_vessel(
    label_great_vessel, label_ventricle, initial_dilation=(4, 4, 5), final_erosion=(3, 3, 3)
):
    """
    Generates a geometrically-defined valve.
    This function is suitable for the pulmonic and aortic valves.

    Args:
        label_great_vessel (SimpleITK.Image): The binary mask for the great vessel
            (pulmonary artery or ascending aorta)
        label_ventricle (SimpleITK.Image): The binary mask for the ventricle (left or right)
        initial_dilation (tuple, optional): Initial dilation, larger values increase the valve
            size. Defaults to (4, 4, 5).
        final_erosion (tuple, optional): Final erosion, larger values decrease the valve size.
            Defaults to (3, 3, 3).

    Returns:
        SimpleITK.Image: The geometric valve, as a binary mask.
    """
    # Dilate the great vessel and ventricle
    label_great_vessel_dilate = sitk.BinaryDilate(label_great_vessel, initial_dilation)
    label_ventricle_dilate = sitk.BinaryDilate(label_ventricle, initial_dilation)

    # Find the overlap (of these dilated volumes)
    overlap = label_great_vessel_dilate & label_ventricle_dilate

    # Create a mask, first we calculate the union
    dilation = 1
    union_vol = 0
    while union_vol <= 2000:
        union = sitk.BinaryDilate(label_great_vessel, (dilation,) * 3) | sitk.BinaryDilate(
            label_ventricle, (dilation,) * 3
        )
        union_vol = np.sum(sitk.GetArrayFromImage(union) * np.product(union.GetSpacing()))
        dilation += 1

    mask = sitk.Mask(overlap, union)

    label_valve = sitk.BinaryMorphologicalClosing(mask)
    label_valve = sitk.BinaryErode(label_valve, final_erosion)

    return label_valve


def generate_valve_using_cylinder(
    label_atrium,
    label_ventricle,
    label_wh,
    radius_mm=15,
    height_mm=10,
    shift_parameters=[
        [7.63383999e-01, -1.15883572e00, 2.12311297e00],
        [4.21062525e-03, -3.95014189e-04, 1.13108043e-03],
    ],
):
    """
    Generates a geometrically-defined valve.
    This function is suitable for the tricuspid and mitral valves.

    Note: the shift parameters have been determined empirically.
    For the mitral valve, use the defaults.
    For the tricuspid valve, use np.zeros((2,3))

    Args:
        label_atrium (SimpleITK.Image): The binary mask for the (left or right) atrium.
        label_ventricle (SimpleITK.Image): The binary mask for the (left or right) ventricle.
        label_wh (SimpleITK.Image): The binary mask for the whole heart. Used to scale the shift.
        radius_mm (int, optional): The valve radius, in mm. Defaults to 15.
        height_mm (int, optional): The valve height (i.e. perpendicular extend), in mm.
            Defaults to 10.
        shift_parameters (list, optional):
            Shift parameters, which are the intercept (first row) and gradient (second row)
                of a linear function that maps whole heart volume to 3D shift
                (axial, coronal, sagittal). Set to zero to not use.
                Defaults to
                    [ [7.63383999e-01, -1.15883572e00, 2.12311297e00],
                      [4.21062525e-03, -3.95014189e-04, 1.13108043e-03], ].

    Returns:
        SimpleITK.Image: The geometrically defined valve
    """
    # Define the overlap region (using binary dilation)
    # Increment overlap to make sure we have enough voxels
    dilation = 1
    overlap_vol = 0
    while overlap_vol <= 10000:
        overlap = sitk.BinaryDilate(label_atrium, (dilation,) * 3) & sitk.BinaryDilate(
            label_ventricle, (dilation,) * 3
        )
        overlap_vol = np.sum(sitk.GetArrayFromImage(overlap) * np.product(overlap.GetSpacing()))
        dilation += 1

    com_overlap = get_com(overlap, as_int=False)

    # Use empirical model to shift
    wh_vol = sitk.GetArrayFromImage(label_wh).sum() * np.product(label_wh.GetSpacing()) / 1000
    shift = np.dot([1, wh_vol], shift_parameters)
    com_overlap_shifted = np.array(com_overlap) - shift

    # Create a small expanded overlap region
    overlap = sitk.BinaryDilate(label_atrium, (1,) * 3) & sitk.BinaryDilate(
        label_ventricle, (1,) * 3
    )

    # Find the point in this small overlap region closest to the shifted location
    separation_vector_pixels = (
        np.stack(np.where(sitk.GetArrayFromImage(overlap))) - com_overlap_shifted[:, None]
    ) ** 2
    spacing = np.array(label_atrium.GetSpacing())
    separation_vector_mm = separation_vector_pixels / spacing[:, None]

    separation_mm = np.sum(separation_vector_mm, axis=0)
    closest_overlap_point = np.argmin(separation_mm)

    # Now we can calculate the location of the valve
    valve_loc = np.stack(np.where(sitk.GetArrayFromImage(overlap)))[:, closest_overlap_point]
    valve_loc_real = label_ventricle.TransformContinuousIndexToPhysicalPoint(
        valve_loc.tolist()[::-1]
    )

    # Now we create a cylinder with the user_defined parameters
    cylinder = insert_cylinder_image(0 * label_ventricle, radius_mm, height_mm, valve_loc[::-1])

    # Now we compute the first principal moment (long axis) of the larger chamber (2)
    # f = sitk.LabelShapeStatisticsImageFilter()
    # f.Execute(label_ventricle)
    # orientation_vector = f.GetPrincipalAxes(1)

    # A more robust method is to use the COM offset from the chambers
    # as a proxy for the long axis of the LV/RV
    # orientation_vector = np.array(get_com(label_ventricle, real_coords=True)) - np.array(
    #     get_com(label_atrium, real_coords=True)
    # )

    # Another method is to compute the third principal moment of the overlap region
    f = sitk.LabelShapeStatisticsImageFilter()
    f.Execute(overlap)
    orientation_vector = f.GetPrincipalAxes(1)[:3]

    # Get the rotation parameters
    rotation_angle = vector_angle(orientation_vector, (0, 0, 1))
    rotation_axis = np.cross(orientation_vector, (0, 0, 1))

    # Rotate the cylinder to define the valve
    label_valve = rotate_image(
        cylinder,
        rotation_centre=valve_loc_real,
        rotation_axis=rotation_axis,
        rotation_angle_radians=rotation_angle,
        interpolation=sitk.sitkNearestNeighbor,
        default_value=0,
    )

    # Now we want to trim any parts of the valve too close to the edge of the chambers
    # combined_chambers = sitk.BinaryDilate(label_atrium, (3,) * 3) | sitk.BinaryDilate(
    #     label_ventricle, (3,) * 3
    # )
    # combined_chambers = sitk.BinaryErode(combined_chambers, (6, 6, 6))

    # label_valve = sitk.Mask(label_valve, combined_chambers)

    return label_valve
