"""
Deformatoin field operations
"""

import SimpleITK as sitk
import numpy as np

from skimage.morphology import convex_hull_image

from platipy.imaging.registration.registration import (
    apply_field,
    fast_symmetric_forces_demons_registration,
    convert_mask_to_reg_structure,
)


def get_bone_mask(image, lower_threshold=350, upper_threshold=3500, max_hole_size=5):
    """
    Automatically generate a binary mask of bones from a CT image.

    Args:
        image ([SimpleITK.Image]): The patient x-ray CT image to segment.
        lower_threshold (int, optional): Lower voxel value for threshold. Defaults to 350.
        upper_threshold (int, optional): Upper voxel value for threshold. Defaults to 3500.
        max_hole_size (int | list | bool, optional): Maximum hole size to be filled in millimetres. Can be specified as a vector (z,y,x). Defaults to 5.

    Returns:
        [SimpleITK.Image]: The binary bone mask.
    """

    bone_mask = sitk.BinaryThreshold(
        image, lowerThreshold=lower_threshold, upperThreshold=upper_threshold
    )

    if max_hole_size is not False:
        if not hasattr(max_hole_size, "__iter__"):
            max_hole_size = (max_hole_size,) * 3

    bone_mask = sitk.BinaryMorphologicalClosing(bone_mask, max_hole_size)

    return bone_mask


def get_external_mask(
    image, lower_threshold=-100, upper_threshold=2500, dilate=1, max_hole_size=False
):
    """
    Automatically generate a binary mask of the patient external contour.
    Uses slice-wise convex hull generation.

    Args:
        image ([SimpleITK.Image]): The patient x-ray CT image to segment. May work with other modalities with modified thresholds.
        lower_threshold (int, optional): Lower voxel value for threshold. Defaults to -100.
        upper_threshold (int, optional): Upper voxel value for threshold. Defaults to 2500.
        dilate (int | list | bool, optional): Dilation filter size applied to the binary mask. Can be specified as a vector (z,y,x). Defaults to 1.
        max_hole_size (int  | list | bool, optional): Maximum hole size to be filled in millimetres. Can be specified as a vector (z,y,x). Defaults to False.

    Returns:
        [SimpleITK.Image]: The binary external mask.
    """

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


def generate_field_shift(mask_image, vector_shift=(10, 10, 10), gaussian_smooth=5):
    """
    Shifts (moves) a structure defined using a binary mask.

    Args:
        mask_image ([SimpleITK.Image]): The binary mask to shift.
        vector_shift (tuple, optional): The displacement vector applied to the entire binary mask.
                                        Convention: (+/-, +/-, +/-) = (sup/inf, post/ant, left/right) shift.
                                        Defined in millimetres.
                                        Defaults to (10, 10, 10).
        gaussian_smooth (int | list, optional): Scale of a Gaussian kernel used to smooth the deformation vector field. Defaults to 5.

    Returns:
        [SimpleITK.Image]: The binary mask following the shift.
        [SimpleITK.DisplacementFieldTransform]: The transform representing the shift.
        [SimpleITK.Image]: The displacement vector field representing the shift.
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

    return mask_image_shift, dvf_tfm, dvf_template


def generate_field_asymmetric_contract(
    mask_image, vector_asymmetric_contract=(10, 10, 10), gaussian_smooth=5
):
    """
    Contracts a structure (defined using a binary mask) using a specified vector.

    Args:
        mask_image ([SimpleITK.Image]): The binary mask to contract.
        vector_asymmetric_contract (tuple, optional): The contraction vector applied to the entire binary mask.
                                                      Convention: (+/-, +/-, +/-) = (sup/inf, post/ant, left/right) border is contracted.
                                                      Defined in millimetres.
                                                      Defaults to (10, 10, 10).
        gaussian_smooth (int | list, optional): Scale of a Gaussian kernel used to smooth the deformation vector field. Defaults to 5.

    Returns:
        [SimpleITK.Image]: The binary mask following the contract.
        [SimpleITK.DisplacementFieldTransform]: The transform representing the contract.
        [SimpleITK.Image]: The displacement vector field representing the contract.
    """
    # Define array
    # Used for image array manipulations
    mask_image_arr = sitk.GetArrayFromImage(mask_image)

    # The template deformation field
    # Used to generate transforms
    dvf_arr = np.zeros(mask_image_arr.shape + (3,))
    dvf_arr = dvf_arr + np.array([[[vector_asymmetric_contract[::-1]]]])
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

    return mask_image_asymmetric_contract, dvf_tfm, dvf_template


def generate_field_asymmetric_extend(
    mask_image, vector_asymmetric_extend=(10, 10, 10), gaussian_smooth=5
):
    """
    Extends a structure (defined using a binary mask) using a specified vector.

    Args:
        mask_image ([SimpleITK.Image]): The binary mask to extend.
        vector_asymmetric_extend (tuple, optional): The extension vector applied to the entire binary mask.
                                                    Convention: (+/-, +/-, +/-) = (sup/inf, post/ant, left/right) border is extended.
                                                    Defined in millimetres.
                                                    Defaults to (10, 10, 10).
        gaussian_smooth (int | list, optional): Scale of a Gaussian kernel used to smooth the deformation vector field. Defaults to 5.

    Returns:
        [SimpleITK.Image]: The binary mask following the extension.
        [SimpleITK.DisplacementFieldTransform]: The transform representing the extension.
        [SimpleITK.Image]: The displacement vector field representing the extension.
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

    return mask_image_asymmetric_extend, dvf_tfm, dvf_template


def generate_field_expand(
    mask_image,
    bone_mask=False,
    expand=3,
    gaussian_smooth=5,
):
    """
    Expands a structure (defined using a binary mask) using a specified vector to define the dilation kernel.

    Args:
        mask_image ([SimpleITK.Image]): The binary mask to expand.
        bone_mask ([SimpleITK.Image, optional]): A binary mask defining regions where we expect restricted deformations.
        vector_asymmetric_extend (int |tuple, optional): The expansion vector applied to the entire binary mask.
                                                    Convention: (z,y,x) size of expansion kernel.
                                                    Defined in millimetres.
                                                    Defaults to 3.
        gaussian_smooth (int | list, optional): Scale of a Gaussian kernel used to smooth the deformation vector field. Defaults to 5.

    Returns:
        [SimpleITK.Image]: The binary mask following the expansion.
        [SimpleITK.DisplacementFieldTransform]: The transform representing the expansion.
        [SimpleITK.Image]: The displacement vector field representing the expansion.
    """

    registration_mask_original = convert_mask_to_reg_structure(mask_image_original)

    if bone_mask is not False:
        mask_image_original = mask_image + bone_mask
    else:
        mask_image_original = mask_image

    # Use binary erosion to create a smaller volume
    if not hasattr(expand, "__iter__"):
        expand = (expand,) * 3

    expand = np.array(expand)

    # Convert voxels to millimetres
    expand = expand / np.array(mask_image.GetSpacing()[::-1])

    # Re-order to (x,y,z)
    expand = expand[::-1]
    # expand = [int(i / j) for i, j in zip(expand, mask_image.GetSpacing()[::-1])][::-1]

    # If all negative: erode
    if np.all(np.array(expand) <= 0):
        print("All factors negative: shrinking only.")
        mask_image_expand = sitk.BinaryErode(
            mask_image, np.abs(expand).astype(int).tolist(), sitk.sitkBall
        )

    # If all positive: dilate
    elif np.all(np.array(expand) >= 0):
        print("All factors positive: expansion only.")
        mask_image_expand = sitk.BinaryDilate(
            mask_image, np.abs(expand).astype(int).tolist(), sitk.sitkBall
        )

    # Otherwise: sequential operations
    else:
        print("Mixed factors: shrinking and expansion.")
        expansion_kernel = expand * (expand > 0)
        shrink_kernel = expand * (expand < 0)

        mask_image_expand = sitk.BinaryDilate(
            mask_image, np.abs(expansion_kernel).astype(int).tolist(), sitk.sitkBall
        )
        mask_image_expand = sitk.BinaryErode(
            mask_image_expand, np.abs(shrink_kernel).astype(int).tolist(), sitk.sitkBall
        )

    registration_mask_expand = convert_mask_to_reg_structure(mask_image_expand)
    if bone_mask is not False:
        registration_mask_expand = registration_mask_expand + bone_mask

    # Use DIR to find the deformation
    _, _, dvf_template = fast_symmetric_forces_demons_registration(
        registration_mask_expand,
        registration_mask_original,
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

    return mask_image_symmetric_expand, dvf_tfm, dvf_template


def generate_field_radial_bend(
    reference_image,
    body_mask,
    reference_point,
    axis_of_rotation=[0, 0, -1],
    scale=0.1,
    mask_bend_from_reference_point=("z", "inf"),
    gaussian_smooth=5,
):
    """
    Generates a synthetic field characterised by radial bending.
    Typically, this field would be used to simulate a moving head and so masking is important.

    Args:
        reference_image ([SimpleITK.Image]): The image to be deformed.
        body_mask ([SimpleITK.Image]): A binary mask in which the deformation field will be defined
        reference_point ([tuple]): The point (z,y,x) about which the rotation field is defined.
        axis_of_rotation (tuple, optional): The axis of rotation (z,y,x). Defaults to [0, 0, -1].
        scale (int, optional): The deformation vector length at each point will equal scale multiplied by the distance to that point from reference_point. Defaults to 1.
        mask_bend_from_reference_point (tuple, optional): The dimension (z=axial, y=coronal, x=sagittal) and limit (inf/sup, post/ant, left/right) for masking the vector field, relative to the reference point. Defaults to ("z", "inf").
        gaussian_smooth (int | list, optional): Scale of a Gaussian kernel used to smooth the deformation vector field. Defaults to 5.

    Returns:
        [SimpleITK.Image]: The binary mask following the expansion.
        [SimpleITK.DisplacementFieldTransform]: The transform representing the expansion.
        [SimpleITK.Image]: The displacement vector field representing the expansion.
    """

    body_mask_arr = sitk.GetArrayFromImage(body_mask)

    if mask_bend_from_reference_point is not False:
        if mask_bend_from_reference_point[0] == "z":
            if mask_bend_from_reference_point[1] == "inf":
                body_mask_arr[: reference_point[0], :, :] = 0
            elif mask_bend_from_reference_point[1] == "sup":
                body_mask_arr[reference_point[0] :, :, :] = 0
        if mask_bend_from_reference_point[0] == "y":
            if mask_bend_from_reference_point[1] == "post":
                body_mask_arr[:, reference_point[1] :, :] = 0
            elif mask_bend_from_reference_point[1] == "ant":
                body_mask_arr[:, : reference_point[1], :] = 0
        if mask_bend_from_reference_point[0] == "x":
            if mask_bend_from_reference_point[1] == "left":
                body_mask_arr[:, :, reference_point[2] :] = 0
            elif mask_bend_from_reference_point[1] == "right":
                body_mask_arr[:, :, : reference_point[2]] = 0

    pt_arr = np.array(np.where(body_mask_arr))
    vector_ref_to_pt = pt_arr - np.array(reference_point)[:, None]

    # Normalise the normal vector (axis_of_rotation)
    axis_of_rotation = np.array(axis_of_rotation)
    axis_of_rotation = axis_of_rotation / np.linalg.norm(axis_of_rotation)

    deformation_vectors = np.cross(vector_ref_to_pt[::-1].T, axis_of_rotation[::-1])

    dvf_template = sitk.Image(reference_image.GetSize(), sitk.sitkVectorFloat64, 3)
    dvf_template_arr = sitk.GetArrayFromImage(dvf_template)

    if scale is not False:
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

    return reference_image_bend, dvf_tfm, dvf_template
