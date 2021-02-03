"""
Deformatoin field operations
"""

import SimpleITK as sitk
import numpy as np

from skimage.morphology import convex_hull_image

from platipy.imaging.registration.registration import (
    apply_field,
    fast_symmetric_forces_demons_registration,
)


def get_bone_mask(image, lower_threshold=350, upper_threshold=3500, max_hole_size=5):
    """[summary]

    Args:
        image ([type]): [description]
        threshold (int, optional): [description]. Defaults to 350.
        max_hole_size (int, optional): [description]. Defaults to 5.
    """

    bone_mask = sitk.BinaryThreshold(
        image, lowerThreshold=lower_threshold, upperThreshold=upper_threshold
    )

    if not hasattr(max_hole_size, "__iter__"):
        max_hole_size = (max_hole_size,) * 3

    bone_mask = sitk.BinaryMorphologicalClosing(bone_mask, max_hole_size)

    return bone_mask


def get_external_mask(
    image, lower_threshold=-100, upper_threshold=2500, dilate=1, max_hole_size=False
):

    # Get all points inside the body
    external_mask = sitk.BinaryThreshold(
        image, lowerThreshold=lower_threshold, upperThreshold=upper_threshold
    )

    external_mask_components = sitk.ConnectedComponent(external_mask, True)

    # Second largest volume is most likely the body - you should check this!
    body_mask = sitk.Equal(sitk.RelabelComponent(external_mask_components), 1)

    if dilate is not False:
        if not hasattr(dilate, "__iter__"):
            dilate = (dilate,) * 3
        body_mask = sitk.BinaryDilate(body_mask, dilate)

    if max_hole_size is not False:
        if not hasattr(max_hole_size, "__iter__"):
            max_hole_size = (max_hole_size,) * 3

        body_mask = sitk.BinaryMorphologicalClosing(body_mask, max_hole_size)
        body_mask = sitk.BinaryFillhole(body_mask, fullyConnected=True)

    arr = sitk.GetArrayFromImage(body_mask)

    convex_hull_slices = np.zeros_like(arr)

    for index in np.arange(0, np.alen(arr)):
        convex_hull_slices[index] = convex_hull_image(arr[index])

    body_mask_hull = sitk.GetImageFromArray(convex_hull_slices)
    body_mask_hull.CopyInformation(body_mask)

    return body_mask_hull


def generate_field_shift(
    mask_image,
    vector_shift=(10, 10, 10),
    gaussian_smooth=5,
    return_mask=True,
    return_transform=True,
):
    """[summary]

    Args:
        mask_image ([type]): [description]
        vector_shift (tuple, optional): Defined as (axial, coronal, sagittal) shift, with signs defining shifts as follows: (+/-, +/-, +/-) = (sup/inf, ant/post, l/r) in patient coordinates.. Defaults to (10,10,10).
        post_shift_gaussian_smooth (tuple, optional): [description]. Defaults to (2,2,2).
        return_mask (bool, optional): [description]. Defaults to True.
        return_transform (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]

    Yields:
        [type]: [description]
    """
    # Define array
    # Used for image array manipulations
    mask_image_arr = sitk.GetArrayFromImage(mask_image)

    # The template deformation field
    # Used to generate transforms
    dvf_arr = np.zeros(mask_image_arr.shape + (3,))
    dvf_arr = dvf_arr - np.array([[[vector_shift[::-1]]]])
    dvf_template = sitk.GetImageFromArray(dvf_arr)

    # Copy image information
    dvf_template.CopyInformation(mask_image)

    dvf_tfm = sitk.DisplacementFieldTransform(
        sitk.Cast(dvf_template, sitk.sitkVectorFloat64)
    )
    mask_image_shift = apply_field(
        mask_image, transform=dvf_tfm, structure=True, interp=1
    )

    dvf_template = sitk.Mask(dvf_template, mask_image | mask_image_shift)

    # smooth
    if np.any(gaussian_smooth):

        if not hasattr(gaussian_smooth, "__iter__"):
            gaussian_smooth = (gaussian_smooth,) * 3

        dvf_template = sitk.SmoothingRecursiveGaussian(dvf_template, gaussian_smooth)

    dvf_tfm = sitk.DisplacementFieldTransform(
        sitk.Cast(dvf_template, sitk.sitkVectorFloat64)
    )
    mask_image_shift = apply_field(
        mask_image, transform=dvf_tfm, structure=True, interp=1
    )

    if return_mask:

        if return_transform:
            return mask_image_shift, dvf_tfm, dvf_template

        else:
            return mask_image_shift, dvf_template

    else:
        return dvf_template


def generate_field_asymmetric_contract(
    mask_image,
    vector_asymmetric_contract=(10, 10, 10),
    gaussian_smooth=5,
    return_mask=True,
    return_transform=True,
):
    """[summary]

    Args:
        mask_image ([type]): [description]
        vector_asymmetric_contract (tuple, optional): Defined as (axial, coronal, sagittal) asymmetric_contract.
            Signs defining asymmetric_contracts as follows: (+/-, +/-, +/-) = contract volume at (inf/sup, post/ant, r/l) border in patient coordinates.
            Defaults to (10,10,10).
        post_asymmetric_contract_gaussian_smooth (tuple, optional): [description]. Defaults to (2,2,2).
        return_mask (bool, optional): [description]. Defaults to True.
        return_transform (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]

    Yields:
        [type]: [description]
    """
    # Define array
    # Used for image array manipulations
    mask_image_arr = sitk.GetArrayFromImage(mask_image)

    # The template deformation field
    # Used to generate transforms
    dvf_arr = np.zeros(mask_image_arr.shape + (3,))
    dvf_arr = dvf_arr - np.array([[[vector_asymmetric_contract[::-1]]]])
    dvf_template = sitk.GetImageFromArray(dvf_arr)

    # Copy image information
    dvf_template.CopyInformation(mask_image)

    dvf_template = sitk.Mask(dvf_template, mask_image)

    dvf_tfm = sitk.DisplacementFieldTransform(
        sitk.Cast(dvf_template, sitk.sitkVectorFloat64)
    )
    mask_image_asymmetric_contract = apply_field(
        mask_image, transform=dvf_tfm, structure=True, interp=1
    )

    # smooth
    if np.any(gaussian_smooth):

        if not hasattr(gaussian_smooth, "__iter__"):
            gaussian_smooth = (gaussian_smooth,) * 3

        dvf_template = sitk.SmoothingRecursiveGaussian(dvf_template, gaussian_smooth)

    dvf_tfm = sitk.DisplacementFieldTransform(
        sitk.Cast(dvf_template, sitk.sitkVectorFloat64)
    )
    mask_image_asymmetric_contract = apply_field(
        mask_image, transform=dvf_tfm, structure=True, interp=1
    )

    if return_mask:

        if return_transform:
            return mask_image_asymmetric_contract, dvf_tfm, dvf_template

        else:
            return mask_image_asymmetric_contract, dvf_template

    else:
        return dvf_template


def generate_field_asymmetric_extend(
    mask_image,
    vector_asymmetric_extend=(10, 10, 10),
    gaussian_smooth=5,
    return_mask=True,
    return_transform=True,
):
    """[summary]

    Args:
        mask_image ([type]): [description]
        vector_asymmetric_extend (tuple, optional): Defined as (axial, coronal, sagittal) asymmetric_extend, with signs defining asymmetric_extends as follows: (+/-, +/-, +/-) = (sup/inf, ant/post, l/r) in patient coordinates.. Defaults to (10,10,10).
        post_asymmetric_extend_gaussian_smooth (tuple, optional): [description]. Defaults to (2,2,2).
        return_mask (bool, optional): [description]. Defaults to True.
        return_transform (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]

    Yields:
        [type]: [description]
    """
    # Define array
    # Used for image array manipulations
    mask_image_arr = sitk.GetArrayFromImage(mask_image)

    # The template deformation field
    # Used to generate transforms
    dvf_arr = np.zeros(mask_image_arr.shape + (3,))
    dvf_arr = dvf_arr - np.array([[[vector_asymmetric_extend[::-1]]]])
    dvf_template = sitk.GetImageFromArray(dvf_arr)

    # Copy image information
    dvf_template.CopyInformation(mask_image)

    dvf_tfm = sitk.DisplacementFieldTransform(
        sitk.Cast(dvf_template, sitk.sitkVectorFloat64)
    )
    mask_image_asymmetric_extend = apply_field(
        mask_image, transform=dvf_tfm, structure=True, interp=1
    )

    dvf_template = sitk.Mask(dvf_template, mask_image_asymmetric_extend)

    # smooth
    if np.any(gaussian_smooth):

        if not hasattr(gaussian_smooth, "__iter__"):
            gaussian_smooth = (gaussian_smooth,) * 3

        dvf_template = sitk.SmoothingRecursiveGaussian(dvf_template, gaussian_smooth)

    dvf_tfm = sitk.DisplacementFieldTransform(
        sitk.Cast(dvf_template, sitk.sitkVectorFloat64)
    )
    mask_image_asymmetric_extend = apply_field(
        mask_image, transform=dvf_tfm, structure=True, interp=1
    )

    if return_mask:

        if return_transform:
            return mask_image_asymmetric_extend, dvf_tfm, dvf_template

        else:
            return mask_image_asymmetric_extend, dvf_template

    else:
        return dvf_template


def generate_field_expand(
    mask_image,
    bone_mask=False,
    expand=3,
    gaussian_smooth=5,
    return_mask=True,
    return_transform=True,
):

    if bone_mask is not False:
        mask_image_original = mask_image + bone_mask
    else:
        mask_image_original = mask_image

    # Use binary erosion to create a smaller volume
    if not hasattr(expand, "__iter__"):
        expand = (expand,) * 3

    if np.all(np.array(expand) <= 0):
        mask_images_expand = sitk.BinaryErode(mask_image, list(np.abs(expand)))
    elif np.all(np.array(expand) >= 0):
        mask_images_expand = sitk.BinaryDilate(mask_image, expand)
    else:
        raise ValueError(
            "All values for expand should be the same sign (positive = expand, negative = shrink)."
        )

    if bone_mask is not False:
        mask_images_expand = mask_images_expand + bone_mask

    # Use DIR to find the deformation
    _, dvf_tfm, dvf_template = fast_symmetric_forces_demons_registration(
        mask_images_expand,
        mask_image_original,
        isotropic_resample=True,
        resolution_staging=[4, 2],
        iteration_staging=[10, 10],
        ncores=8,
        return_field=True,
    )

    # smooth
    if np.any(gaussian_smooth):

        if not hasattr(gaussian_smooth, "__iter__"):
            gaussian_smooth = (gaussian_smooth,) * 3

        dvf_template = sitk.SmoothingRecursiveGaussian(dvf_template, gaussian_smooth)

    dvf_tfm = sitk.DisplacementFieldTransform(
        sitk.Cast(dvf_template, sitk.sitkVectorFloat64)
    )
    mask_image_symmetric_expand = apply_field(
        mask_image, transform=dvf_tfm, structure=True, interp=1
    )

    if return_mask:

        if return_transform:
            return mask_image_symmetric_expand, dvf_tfm, dvf_template

        else:
            return mask_image_symmetric_expand, dvf_template

    else:
        return dvf_template


def generate_field_radial_bend(
    reference_image,
    body_mask,
    reference_point,
    normal_vector=[0, 0, -1],
    # field_mask,
    scale=1,
    mask_bend_from_reference_point=("z", "inf"),
    gaussian_smooth=5,
    return_mask=True,
    return_transform=True,
):

    # Get an array from either the mask
    # if field_mask is not False:
    #     field_arr = sitk.GetArrayFromImage(field_arr)
    # else:
    #     field_arr = ()

    body_mask_arr = sitk.GetArrayFromImage(body_mask)

    if mask_bend_from_reference_point is not False:
        if mask_bend_from_reference_point[0] == "z":
            if mask_bend_from_reference_point[1] == "inf":
                body_mask_arr[: reference_point[0], :, :] = 0
            elif mask_bend_from_reference_point[1] == "sup":
                body_mask_arr[reference_point[0] :, :, :] = 0
        if mask_bend_from_reference_point[0] == "y":
            if mask_bend_from_reference_point[1] == "pos":
                body_mask_arr[:, reference_point[1] :, :] = 0
            elif mask_bend_from_reference_point[1] == "ant":
                body_mask_arr[:, : reference_point[1], :] = 0
        if mask_bend_from_reference_point[0] == "x":
            if mask_bend_from_reference_point[1] == "left":
                body_mask_arr[:, :, reference_point[2] :] = 0
            elif mask_bend_from_reference_point[1] == "right":
                body_mask_arr[:, :, : reference_point[2]] = 0

    ####
    # pt_arr = np.array(np.where(body_mask_arr))
    # vector_ref_to_pt = pt_arr - np.array(larynx_reference_point[::-1])[:,None]

    # print('  Generating vectors')

    # deformation_vectors = np.cross(vector_ref_to_pt[::-1].T, [-1,0,0])

    # dvf_arr[np.where(body_mask_arr)] = deformation_vectors / 3
    ####

    pt_arr = np.array(np.where(body_mask_arr))
    vector_ref_to_pt = pt_arr - np.array(reference_point)[:, None]

    print("  Generating vectors")

    deformation_vectors = np.cross(vector_ref_to_pt[::-1].T, normal_vector[::-1])

    dvf_template = sitk.Image(reference_image.GetSize(), sitk.sitkVectorFloat64, 3)
    dvf_template_arr = sitk.GetArrayFromImage(dvf_template)
    dvf_template_arr[np.where(body_mask_arr)] = deformation_vectors * scale

    dvf_template = sitk.GetImageFromArray(dvf_template_arr)
    dvf_template.CopyInformation(reference_image)

    # smooth
    if np.any(gaussian_smooth):

        if not hasattr(gaussian_smooth, "__iter__"):
            gaussian_smooth = (gaussian_smooth,) * 3

        dvf_template = sitk.SmoothingRecursiveGaussian(dvf_template, gaussian_smooth)

    dvf_tfm = sitk.DisplacementFieldTransform(
        sitk.Cast(dvf_template, sitk.sitkVectorFloat64)
    )
    reference_image_bend = apply_field(
        reference_image,
        transform=dvf_tfm,
        structure=False,
        default_value=int(sitk.GetArrayViewFromImage(reference_image).min()),
        interp=2,
    )

    if return_mask:

        if return_transform:
            return reference_image_bend, dvf_tfm, dvf_template

        else:
            return reference_image_bend, dvf_template

    else:
        return dvf_template
