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

from platipy.imaging.utils.crop import crop_to_roi, label_to_roi


def generate_valve_from_great_vessel(
    label_great_vessel,
    label_ventricle,
    valve_thickness_mm=8,
):
    """
    Generates a geometrically-defined valve.
    This function is suitable for the pulmonic and aortic valves.

    Args:
        label_great_vessel (SimpleITK.Image): The binary mask for the great vessel
            (pulmonary artery or ascending aorta)
        label_ventricle (SimpleITK.Image): The binary mask for the ventricle (left or right)
        valve_thickness_mm (int, optional): Valve thickness, in millimetres. Defaults to 8.

    Returns:
        SimpleITK.Image: The geometric valve, as a binary mask.
    """

    # To speed up binary morphology operations we first crop all images
    template_img = 0 * label_ventricle
    cb_size, cb_index = label_to_roi(
        (label_great_vessel + label_ventricle) > 0, expansion_mm=(20, 20, 20)
    )

    label_ventricle = crop_to_roi(label_ventricle, cb_size, cb_index)
    label_great_vessel = crop_to_roi(label_great_vessel, cb_size, cb_index)

    # Convert valve thickness to voxels
    _, _, res_z = label_ventricle.GetSpacing()
    valve_thickness = int(valve_thickness_mm / res_z)

    # Dilate the ventricle
    label_ventricle_dilate = sitk.BinaryDilate(label_ventricle, (valve_thickness,) * 3)

    # Find the overlap
    overlap = label_great_vessel & label_ventricle_dilate

    # Mask to thinner great vessel
    mask = label_great_vessel | label_ventricle_dilate

    overlap = sitk.Mask(overlap, mask)

    label_valve = sitk.BinaryMorphologicalClosing(overlap)

    # Finally, paste back to the original image space
    label_valve = sitk.Paste(
        template_img,
        label_valve,
        label_valve.GetSize(),
        (0, 0, 0),
        cb_index,
    )

    return label_valve


def generate_valve_using_cylinder(
    label_atrium,
    label_ventricle,
    radius_mm=15,
    height_mm=10,
):
    """
    Generates a geometrically-defined valve.
    This function is suitable for the tricuspid and mitral valves.

    Args:
        label_atrium (SimpleITK.Image): The binary mask for the (left or right) atrium.
        label_ventricle (SimpleITK.Image): The binary mask for the (left or right) ventricle.
        radius_mm (int, optional): The valve radius, in mm. Defaults to 15.
        height_mm (int, optional): The valve height (i.e. perpendicular extent), in mm.
            Defaults to 10.

    Returns:
        SimpleITK.Image: The geometrically defined valve
    """
    # To speed up binary morphology operations we first crop all images
    template_img = 0 * label_ventricle
    cb_size, cb_index = label_to_roi(
        (label_atrium + label_ventricle) > 0, expansion_mm=(20, 20, 20)
    )

    label_atrium = crop_to_roi(label_atrium, cb_size, cb_index)
    label_ventricle = crop_to_roi(label_ventricle, cb_size, cb_index)

    # Define the overlap region (using binary dilation)
    # Increment overlap to make sure we have enough voxels
    dilation = 1
    overlap_vol = 0
    while overlap_vol <= 2000:
        dilation_img = [int(dilation / i) for i in label_ventricle.GetSpacing()]
        overlap = sitk.BinaryDilate(label_atrium, dilation_img) & sitk.BinaryDilate(
            label_ventricle, dilation_img
        )
        overlap_vol = np.sum(sitk.GetArrayFromImage(overlap) * np.product(overlap.GetSpacing()))
        dilation += 1

    # Now we can calculate the location of the valve
    valve_loc = get_com(overlap, as_int=True)
    valve_loc_real = get_com(overlap, real_coords=True)

    # Now we create a cylinder with the user_defined parameters
    cylinder = insert_cylinder_image(0 * label_ventricle, radius_mm, height_mm, valve_loc[::-1])

    # Now we compute the first principal moment (long axis) of the combined chambers
    # f = sitk.LabelShapeStatisticsImageFilter()
    # f.Execute(label_ventricle + label_atrium)
    # orientation_vector = f.GetPrincipalAxes(1)[:3]

    # A more robust method is to use the COM offset from the chambers
    # as a proxy for the long axis of the LV/RV
    orientation_vector = np.array(get_com(label_ventricle, real_coords=True)) - np.array(
        get_com(label_atrium, real_coords=True)
    )

    # Another method is to compute the third principal moment of the overlap region
    # f = sitk.LabelShapeStatisticsImageFilter()
    # f.Execute(overlap)
    # orientation_vector = f.GetPrincipalAxes(1)[:3]

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

    # Finally, paste back to the original image space
    label_valve = sitk.Paste(
        template_img,
        label_valve,
        label_valve.GetSize(),
        (0, 0, 0),
        cb_index,
    )

    return label_valve
