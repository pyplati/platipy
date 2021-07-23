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

from platipy.imaging.utils.crop import crop_to_roi, label_to_roi


def get_closest_point_2d(reference_label, measurement_label):
    """Finds the point on "measurement_label" that is closest to "reference_label"

    Args:
        reference_label (SimpleITK.Image): The reference label, from which distance is measured.
        measurement_label (SimpleITK.Image): The measurement label. We evaluate the distance at
            every location in this volume.

    Returns:
        tuple: The closest point in 2 dimensions.
    """

    # Compute the distance map
    distancemap_2d = sitk.SignedMaurerDistanceMap(
        reference_label, squaredDistance=False, useImageSpacing=True
    )

    # Get the distance from the reference to each point in the measurement label
    arr_distancemap = sitk.GetArrayFromImage(distancemap_2d)
    arr_measurement = sitk.GetArrayFromImage(measurement_label)

    yloc, xloc = np.where(arr_measurement)
    distances = arr_distancemap[yloc, xloc]

    # Find where the distance is a minimum
    location_of_min = distances.argmin()
    yloc, xloc = yloc[location_of_min], xloc[location_of_min]

    return yloc, xloc


def geometric_sinoatrialnode(label_svc, label_ra, label_wholeheart, radius_mm=10):
    """Geometric definition of the cardiac sinoatrial node (SAN).
    This is largely inspired by Loap et al 2021 [https://doi.org/10.1016/j.prro.2021.02.002]

    Args:
        label_svc (SimpleITK.Image): The binary mask defining the superior vena cava.
        label_ra (SimpleITK.Image): The binary mask defining the right atrium.
        label_wholeheart (SimpleITK.Image): The binary mask defining the whole heart.
        radius_mm (int, optional): The radius of the SAN, in millimetres. Defaults to 10.

    Returns:
        SimpleITK.Image: A binary mask defining the SAN
    """

    # To speed up binary morphology operations we first crop all images
    template_img = 0 * label_wholeheart
    cb_size, cb_index = label_to_roi(
        (label_svc + label_ra + label_wholeheart) > 0, expansion_mm=(20, 20, 20)
    )

    label_svc = crop_to_roi(label_svc, cb_size, cb_index)
    label_ra = crop_to_roi(label_ra, cb_size, cb_index)
    label_wholeheart = crop_to_roi(label_wholeheart, cb_size, cb_index)

    arr_svc = sitk.GetArrayFromImage(label_svc)
    arr_ra = sitk.GetArrayFromImage(label_ra)

    # First, find the most inferior slice of the SVC
    # This defines the z location of the SAN
    inf_limit_svc = np.min(np.where(arr_svc)[0])

    # Now expand the SVC until it touches the RA on the inf slice
    overlap = 0
    dilate = 1
    dilate_ax = 0
    while overlap == 0:
        label_svc_dilate = sitk.BinaryDilate(label_svc, (dilate, dilate, dilate_ax))
        label_overlap = label_svc_dilate & label_ra
        overlap = sitk.GetArrayFromImage(label_overlap)[inf_limit_svc, :, :].sum()
        dilate += 1

        if dilate >= 3:
            arr_svc = sitk.GetArrayFromImage(label_svc_dilate)
            inf_limit_svc = np.min(np.where(arr_svc)[0])
            dilate_ax += 1

    # Locate the point on intersection
    intersect_loc = get_com(label_overlap)

    # Create an image with a single voxel of value 1 located at the point of intersection
    arr_intersect = arr_ra * 0
    arr_intersect[inf_limit_svc, intersect_loc[1], intersect_loc[2]] = 1
    label_intersect = sitk.GetImageFromArray(arr_intersect)
    label_intersect.CopyInformation(label_ra)

    # Define the locations greater than 10mm from the WH
    # Ensures the SAN doesn't extend outside the heart volume
    potential_san_region = sitk.BinaryErode(label_wholeheart, (10, 10, 0))

    # Find the point in this region closest to the intersection
    # First generate a distance map
    distancemap_san = sitk.SignedMaurerDistanceMap(
        label_intersect, squaredDistance=False, useImageSpacing=True
    )

    # Then get the distance from the intersection at all possible points
    arr_distancemap_san = sitk.GetArrayFromImage(distancemap_san)
    arr_potential_san_region = sitk.GetArrayFromImage(potential_san_region)

    yloc, xloc = np.where(arr_potential_san_region[inf_limit_svc, :, :])

    distances = arr_distancemap_san[inf_limit_svc, yloc, xloc]

    # Find where the distance is a minimum
    location_of_min = distances.argmin()

    # Now define the SAN location
    sphere_centre = (inf_limit_svc, yloc[location_of_min], xloc[location_of_min])

    # Generate an image
    label_san = insert_sphere_image(label_ra * 0, sp_radius=radius_mm, sp_centre=sphere_centre)

    # Finally, paste the label into the original image space
    label_san = sitk.Paste(
        template_img,
        label_san,
        label_san.GetSize(),
        (0, 0, 0),
        cb_index,
    )

    return label_san


def geometric_atrioventricularnode(label_la, label_lv, label_ra, label_rv, radius_mm=10):
    """Geometric definition of the cardiac atrioventricular node (AVN).
    This is largely inspired by Loap et al 2021 [https://doi.org/10.1016/j.prro.2021.02.002]

    Args:
        label_la (SimpleITK.Image): The binary mask defining the left atrium.
        label_lv (SimpleITK.Image): The binary mask defining the left ventricle.
        label_ra (SimpleITK.Image): The binary mask defining the right atrium.
        label_rv (SimpleITK.Image): The binary mask defining the right ventricle.
        radius_mm (float, optional): The radius of the AVN, in millimetres. Defaults to 10.

    Returns:
        SimpleITK.Image: A binary mask defining the AVN
    """

    # To speed up binary morphology operations we first crop all images
    template_img = 0 * label_ra
    cb_size, cb_index = label_to_roi(
        (label_la + label_lv + label_ra + label_rv) > 0, expansion_mm=(20, 20, 20)
    )

    label_la = crop_to_roi(label_la, cb_size, cb_index)
    label_lv = crop_to_roi(label_lv, cb_size, cb_index)
    label_ra = crop_to_roi(label_ra, cb_size, cb_index)
    label_rv = crop_to_roi(label_rv, cb_size, cb_index)

    # First, find the most inferior slice of the left atrium
    arr_la = sitk.GetArrayFromImage(label_la)
    inf_limit_la = np.min(np.where(arr_la)[0])

    # Now progress 1cm in the superior direction
    # This defines the slice of the AVN centre
    slice_loc = int(inf_limit_la + 10 / label_la.GetSpacing()[2])

    # Create 2D images at this slice location
    label_la_2d = label_la[:, :, slice_loc]
    label_lv_2d = label_lv[:, :, slice_loc]
    label_ra_2d = label_ra[:, :, slice_loc]
    label_rv_2d = label_rv[:, :, slice_loc]

    # We now iteratively erode the structures to ensure they do not overlap
    # This ensures we can measure the closest point without any errors
    # LEFT ATRIUM
    overlap = 1
    erode = 1
    while overlap > 0:
        label_lv_2d = sitk.BinaryErode(label_lv_2d, (erode, erode))
        label_overlap = label_lv_2d & label_la_2d
        overlap = sitk.GetArrayFromImage(label_overlap).sum()
        erode += 1

    # LEFT ATRIUM
    overlap = 0
    erode = 1
    while overlap > 0:
        label_la_2d = sitk.BinaryErode(label_la_2d, (erode, erode))
        label_overlap = label_la_2d & label_ra_2d
        overlap = sitk.GetArrayFromImage(label_overlap).sum()
        erode += 1

    # RIGHT ATRIUM
    overlap = 0
    erode = 1
    while overlap > 0:
        label_ra_2d = sitk.BinaryErode(label_ra_2d, (erode, erode))
        label_overlap = label_ra_2d & label_rv_2d
        overlap = sitk.GetArrayFromImage(label_overlap).sum()
        erode += 1

    # RIGHT VENTRICLE
    overlap = 0
    erode = 1
    while overlap > 0:
        label_rv_2d = sitk.BinaryErode(label_rv_2d, (erode, erode))
        label_overlap = label_rv_2d & label_lv_2d
        overlap = sitk.GetArrayFromImage(label_overlap).sum()
        erode += 1

    # Measure closest points
    y_la, x_la = get_closest_point_2d(label_rv_2d, label_la_2d)
    y_lv, x_lv = get_closest_point_2d(label_ra_2d, label_lv_2d)
    y_ra, x_ra = get_closest_point_2d(label_lv_2d, label_ra_2d)
    y_rv, x_rv = get_closest_point_2d(label_la_2d, label_rv_2d)

    # Take the arithmetic mean
    x_location = np.mean((x_la, x_lv, x_ra, x_rv), dtype=int)
    y_location = np.mean((y_la, y_lv, y_ra, y_rv), dtype=int)

    # Now define the AVN location
    sphere_centre = (slice_loc, y_location, x_location)

    # Generate an image
    label_avn = insert_sphere_image(label_ra * 0, sp_radius=radius_mm, sp_centre=sphere_centre)

    # Finally, paste the label into the original image space
    label_avn = sitk.Paste(
        template_img,
        label_avn,
        label_avn.GetSize(),
        (0, 0, 0),
        cb_index,
    )

    return label_avn
