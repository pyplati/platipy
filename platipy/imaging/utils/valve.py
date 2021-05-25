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

from platipy.imaging.generation.image import insert_sphere_image

"""
Generate valves
First example - meeting point of LA/LV (mitral valve)
"""


def generate_valves_from_chambers(img_chamber_1, img_chamber_2, radius_mm=10):

    # Find the mid-point of the chambers
    com_1 = np.array(get_com(img_chamber_1))
    com_2 = np.array(get_com(img_chamber_2))

    midpoint = com_1 + 0.5 * (com_2 - com_1)

    # Define the overlap region (using binary dilation)
    # Increment overlap to make sure we have enough voxels
    dilation = 1
    overlap_vol = 0
    while overlap_vol <= 2000:
        overlap = sitk.BinaryDilate(img_chamber_1, (dilation,) * 3) & sitk.BinaryDilate(
            img_chamber_2, (dilation,) * 3
        )
        overlap_vol = np.sum(sitk.GetArrayFromImage(overlap) * np.product(overlap.GetSpacing()))
        dilation += 1

    print(f"Sufficient overlap found at dilation = {dilation} [V = {overlap_vol/1000:.2f} cm^3]")

    # Find the point in the overlap region closest to the mid-point
    separation_vector_pixels = (
        np.stack(np.where(sitk.GetArrayFromImage(overlap))) - midpoint[:, None]
    ) ** 2
    spacing = np.array(img_chamber_1.GetSpacing())
    separation_vector_mm = separation_vector_pixels / spacing[:, None]

    separation_mm = np.sum(separation_vector_mm, axis=0)
    closest_overlap_point = np.argmin(separation_mm)

    com_valve = np.stack(np.where(sitk.GetArrayFromImage(overlap)))[:, closest_overlap_point]

    # Define the valve as a sphere
    auto_valve = insert_sphere_image(0 * overlap, sp_radius=radius_mm, sp_centre=com_valve)

    return auto_valve


def generate_valves_from_vessels(vessel, thickness_mm=4, erosion_mm=2):

    # Thickness can be defined by (inferior_thickness, superior_thickness),
    # or just total_thickness
    if hasattr(thickness_mm, "__iter__"):
        thickness_inferior, thickness_superior = thickness_mm
    else:
        thickness_inferior, thickness_superior = thickness_mm * 0.5, thickness_mm * 0.5

    # Get the most inferior slice
    arr_vessel = sitk.GetArrayFromImage(vessel)
    inferior_slice = np.where(arr_vessel)[0].min()

    # Get interior and superior limits
    filled_superior_slice = np.ceil(
        np.where(arr_vessel)[0].min() + (thickness_superior / vessel.GetSpacing()[2])
    ).astype(int)
    filled_inferior_slice = np.floor(
        np.where(arr_vessel)[0].min() - (thickness_inferior / vessel.GetSpacing()[2])
    ).astype(int)

    # Create vessel interior
    vessel_interior = sitk.BinaryErode(vessel, (erosion_mm, erosion_mm, 0))
    arr = sitk.GetArrayFromImage(vessel_interior)

    # Erase upper slices
    arr[filled_superior_slice:, :, :] = 0

    # Copy down (using mirroring about inferior slice)
    for s_in, s_out in zip(
        range(filled_inferior_slice, inferior_slice),
        range(inferior_slice, filled_superior_slice)[::-1],
    ):
        arr[s_in, :, :] = arr[s_out, :, :]

    # Define the valve
    auto_valve = sitk.GetImageFromArray(arr)
    auto_valve.CopyInformation(vessel)

    # Post-processing
    # 1. Extend the actual vessel downwards (continued)
    arr_vessel[:inferior_slice, :, :] = arr_vessel[inferior_slice, :, :]
    continued_vessel = sitk.GetImageFromArray(arr_vessel)
    continued_vessel.CopyInformation(vessel)

    # 2. Erode this continued vessel
    continued_vessel = sitk.BinaryErode(continued_vessel, (erosion_mm, erosion_mm, 0))

    # 3. Mask
    auto_valve = sitk.Mask(auto_valve, continued_vessel)

    # 4. Fill small holes
    auto_valve = sitk.BinaryMorphologicalClosing(auto_valve, (1, 1, 1))

    return auto_valve