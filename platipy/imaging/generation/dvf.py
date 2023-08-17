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

from platipy.imaging.registration.utils import (
    apply_transform,
    convert_mask_to_reg_structure,
    smooth_and_resample,
)

from platipy.imaging.registration.deformable import (
    fast_symmetric_forces_demons_registration,
)


def generate_field_shift(mask_image, vector_shift=(10, 10, 10), isotropic_resolution_mm=4):
    """
    Shifts (moves) a structure defined using a binary mask.

    Args:
        mask_image ([SimpleITK.Image]): The binary mask to shift.
        vector_shift (tuple, optional): The displacement vector applied to the entire binary mask.
                                        Convention: (+/-, +/-, +/-) = (sup/inf, post/ant,
                                        left/right) shift.
                                        Defined in millimetres.
                                        Defaults to (10, 10, 10).
        gaussian_smooth (int | list, optional): Scale of a Gaussian kernel used to smooth the
                                                deformation vector field. Defaults to 5.

    Returns:
        # [SimpleITK.Image]: The binary mask following the shift.
        [SimpleITK.DisplacementFieldTransform]: The transform representing the shift.
        # [SimpleITK.Image]: The displacement vector field representing the shift.
    """
    # Create new image space to speed up everything
    mask_image = smooth_and_resample(
        mask_image,
        isotropic_voxel_size_mm=isotropic_resolution_mm,
        interpolator=sitk.sitkNearestNeighbor,
    )
    # Define array
    # Used for image array manipulations
    mask_image_arr = sitk.GetArrayFromImage(mask_image)

    # Convert image units to physical units
    # vector_shift_img = [i / j for i, j in zip(vector_shift, mask_image.GetSpacing()[::-1])]

    # The template deformation field
    # Used to generate transforms
    dvf_arr = np.zeros(mask_image_arr.shape + (3,))
    dvf_arr = dvf_arr + np.array([[[vector_shift[::-1]]]])
    dvf_template = sitk.GetImageFromArray(dvf_arr)

    # Copy image information
    dvf_template.CopyInformation(mask_image)

    # Mask
    dvf_template = sitk.Mask(dvf_template, mask_image)

    # Invert
    dvf_inv = sitk.InvertDisplacementField(dvf_template)

    return sitk.Cast(dvf_inv, sitk.sitkVectorFloat64)


# dvf_tfm = sitk.DisplacementFieldTransform(sitk.Cast(dvf_template, sitk.sitkVectorFloat64))
# mask_image_shift = apply_transform(
#     mask_image, transform=dvf_tfm, default_value=0, interpolator=sitk.sitkNearestNeighbor
# )

# dvf_template = sitk.Mask(dvf_template, mask_image | mask_image_shift)

# # smooth
# if np.any(gaussian_smooth):

#     if not hasattr(gaussian_smooth, "__iter__"):
#         gaussian_smooth = (gaussian_smooth,) * 3

#     dvf_template = sitk.SmoothingRecursiveGaussian(dvf_template, gaussian_smooth)

# dvf_tfm = sitk.DisplacementFieldTransform(sitk.Cast(dvf_template, sitk.sitkVectorFloat64))
# mask_image_shift = apply_transform(
#     mask_image, transform=dvf_tfm, default_value=0, interpolator=sitk.sitkNearestNeighbor
# )

# return mask_image_shift, dvf_tfm, dvf_template


def generate_field_asymmetric_contract(
    mask_image,
    bone_mask=False,
    vector_asymmetric_contract=(10, 10, 10),
    gaussian_smooth=5,
    compute_real_dvf=False,
):
    """
    Contracts a structure (defined using a binary mask) using a specified vector.

    Args:
        mask_image ([SimpleITK.Image]): The binary mask to contract.
        vector_asymmetric_contract (tuple, optional): The contraction vector applied to the entire
                                                      binary mask.
                                                      Convention: (+/-, +/-, +/-) = (sup/inf,
                                                      post/ant, left/right) border is contracted.
                                                      Defined in millimetres.
                                                      Defaults to (10, 10, 10).
        gaussian_smooth (int | list, optional): Scale of a Gaussian kernel used to smooth the
                                                deformation vector field. Defaults to 5.
        compute_real_dvf (bool, optional): If True, the real deformation vector field is computed.
                                           This involves a slower computation. Defaults to False.

    Returns:
        [SimpleITK.Image]: The binary mask following the contract.
        [SimpleITK.DisplacementFieldTransform]: The transform representing the contract.
        [SimpleITK.Image]: The displacement vector field representing the contract.
    """
    # Apply bone masking
    if bone_mask is not False:
        mask_original = mask_image + bone_mask
    else:
        mask_original = mask_image + 0

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

    dvf_tfm = sitk.DisplacementFieldTransform(sitk.Cast(dvf_template, sitk.sitkVectorFloat64))

    mask_image_asymmetric_contract = apply_transform(
        mask_image, transform=dvf_tfm, default_value=0, interpolator=sitk.sitkNearestNeighbor
    )

    if bone_mask is not False:
        mask_image_asymmetric_contract = mask_image_asymmetric_contract + bone_mask

    if compute_real_dvf:

        reg_struct = convert_mask_to_reg_structure(mask_original)  # , expansion=3)
        reg_struct_def = convert_mask_to_reg_structure(
            mask_image_asymmetric_contract
        )  # , expansion=3)

        # Use DSGR to compute the realistic deformation vector field
        _, _, dvf_template = fast_symmetric_forces_demons_registration(
            reg_struct_def,
            reg_struct,
            isotropic_resample=True,
            resolution_staging=[4, 2],
            iteration_staging=[20, 10],
        )

    # smooth
    if np.any(gaussian_smooth):

        if not hasattr(gaussian_smooth, "__iter__"):
            gaussian_smooth = (gaussian_smooth,) * 3

        dvf_template = sitk.SmoothingRecursiveGaussian(dvf_template, gaussian_smooth)

    dvf_tfm = sitk.DisplacementFieldTransform(sitk.Cast(dvf_template, sitk.sitkVectorFloat64))
    mask_image_asymmetric_contract = apply_transform(
        mask_image, transform=dvf_tfm, default_value=0, interpolator=sitk.sitkNearestNeighbor
    )

    return mask_image_asymmetric_contract, dvf_tfm, dvf_template


def generate_field_asymmetric_extend(
    mask_image, vector_asymmetric_extend=(10, 10, 10), gaussian_smooth=5
):
    """
    Extends a structure (defined using a binary mask) using a specified vector.

    Args:
        mask_image ([SimpleITK.Image]): The binary mask to extend.
        vector_asymmetric_extend (tuple, optional): The extension vector applied to the entire
                                                    binary mask.
                                                    Convention: (+/-, +/-, +/-) = (sup/inf,
                                                    post/ant, left/right) border is extended.
                                                    Defined in millimetres.
                                                    Defaults to (10, 10, 10).
        gaussian_smooth (int | list, optional): Scale of a Gaussian kernel used to smooth the
                                                deformation vector field. Defaults to 5.

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

    dvf_tfm = sitk.DisplacementFieldTransform(sitk.Cast(dvf_template, sitk.sitkVectorFloat64))

    mask_image_asymmetric_extend = apply_transform(
        mask_image, transform=dvf_tfm, default_value=0, interpolator=sitk.sitkNearestNeighbor
    )

    dvf_template = sitk.Mask(dvf_template, mask_image_asymmetric_extend)

    # smooth
    if np.any(gaussian_smooth):

        if not hasattr(gaussian_smooth, "__iter__"):
            gaussian_smooth = (gaussian_smooth,) * 3

        dvf_template = sitk.SmoothingRecursiveGaussian(dvf_template, gaussian_smooth)

    dvf_tfm = sitk.DisplacementFieldTransform(sitk.Cast(dvf_template, sitk.sitkVectorFloat64))

    mask_image_asymmetric_extend = apply_transform(
        mask_image, transform=dvf_tfm, default_value=0, interpolator=sitk.sitkNearestNeighbor
    )

    return mask_image_asymmetric_extend, dvf_tfm, dvf_template


def generate_field_expand(
    mask,
    bone_mask=False,
    expansion_mm=3,
    gaussian_smooth=5,
    use_internal_deformation=True,
):
    """
    Expands a structure (defined using a binary mask) using a specified vector to define the
    dilation kernel.

    Args:
        mask ([SimpleITK.Image]): The binary mask to expand.
        bone_mask ([SimpleITK.Image, optional]): A binary mask defining regions where we expect
                                                 restricted deformations.
        vector_asymmetric_extend (int |tuple, optional): The expansion vector applied to the entire
                                                         binary mask.
                                                    Convention: (z,y,x) size of expansion kernel.
                                                    Defined in millimetres.
                                                    Defaults to 3.
        gaussian_smooth (int | list, optional): Scale of a Gaussian kernel used to smooth the
                                                deformation vector field. Defaults to 5.

    Returns:
        [SimpleITK.Image]: The binary mask following the expansion.
        [SimpleITK.DisplacementFieldTransform]: The transform representing the expansion.
        [SimpleITK.Image]: The displacement vector field representing the expansion.
    """

    # crop to ROI
    crop_box_size, crop_box_index = label_to_roi(mask_expand, expansion_mm=30)

    mask_expand_crop = crop_to_roi(mask_expand, size=crop_box_size, index=crop_box_index)
    mask_air_crop = crop_to_roi(mask_air, size=crop_box_size, index=crop_box_index)

    registration_mask_original = convert_mask_to_reg_structure(mask_air_crop)
    registration_mask_expand = convert_mask_to_reg_structure(mask_expand_crop)

    if bone_mask is not False:
        mask_original = mask + bone_mask
    else:
        mask_original = mask

    # Use binary erosion to create a smaller volume
    if not hasattr(expand, "__iter__"):
        expand = (expand,) * 3

    expand = np.array(expand)

    # Convert voxels to millimetres
    expand = expand / np.array(mask.GetSpacing())
    print("Expansion (num. voxels):", np.abs(expand).astype(int).tolist())

    # If all negative: erode
    if np.all(np.array(expand) <= 0):
        print("All factors negative: shrinking only.")
        mask_expand = sitk.BinaryErode(mask, np.abs(expand).astype(int).tolist(), sitk.sitkBall)

    # If all positive: dilate
    elif np.all(np.array(expand) >= 0):
        print("All factors positive: expansion only.")
        mask_expand = sitk.BinaryDilate(mask, np.abs(expand).astype(int).tolist(), sitk.sitkBall)

    # Otherwise: sequential operations
    else:
        print("Mixed factors: shrinking and expansion.")
        expansion_kernel = expand * (expand > 0)
        shrink_kernel = expand * (expand < 0)

        mask_expand = sitk.BinaryDilate(
            mask, np.abs(expansion_kernel).astype(int).tolist(), sitk.sitkBall
        )
        mask_expand = sitk.BinaryErode(
            mask_expand, np.abs(shrink_kernel).astype(int).tolist(), sitk.sitkBall
        )

    if bone_mask is not False:
        mask_expand = mask_expand + bone_mask

    if use_internal_deformation:
        registration_mask_original = convert_mask_to_reg_structure(mask_original)
        registration_mask_expand = convert_mask_to_reg_structure(mask_expand)

    else:
        registration_mask_original = mask_original
        registration_mask_expand = mask_expand

    # Use DIR to find the deformation
    _, _, dvf_template = fast_symmetric_forces_demons_registration(
        registration_mask_expand,
        registration_mask_original,
        isotropic_resample=True,
        resolution_staging=[4, 2],
        iteration_staging=[20, 20],
        ncores=8,
    )

    # smooth
    if np.any(gaussian_smooth):

        if not hasattr(gaussian_smooth, "__iter__"):
            gaussian_smooth = (gaussian_smooth,) * 3

        dvf_template = sitk.SmoothingRecursiveGaussian(dvf_template, gaussian_smooth)

    dvf_tfm = sitk.DisplacementFieldTransform(sitk.Cast(dvf_template, sitk.sitkVectorFloat64))

    mask_symmetric_expand = apply_transform(
        mask, transform=dvf_tfm, default_value=0, interpolator=sitk.sitkNearestNeighbor
    )

    return mask_symmetric_expand, dvf_tfm, dvf_template


def generate_normal_vectors(
    reference_image, reference_point, axis_of_rotation, angle, exact=False
):
    """
    Generates normal vectors for all points in a mask.
    Utility function used for generating rotational DVFs.

    Args:
        reference_image (SimpleITK.Image): The reference image
        reference_point (tuple): The point (z,y,x) about which the rotation field is defined.
        axis_of_rotation (tuple): The axis of rotation (z,y,x).
        angle (float): The angle of rotation in degrees.

    Returns:
        [SimpleITK.DisplacementFieldTransform]: The transform representing the expansion.
        [SimpleITK.Image]: The displacement vector field representing the expansion.
    """

    # locate all the points in the image (where deformations are defined)
    arr = sitk.GetArrayFromImage(reference_image)
    pt_arr = np.array(np.where(np.isfinite(arr)))

    # define the radial vector
    vector_ref_to_pt = pt_arr - np.array(reference_point)[:, None]
    vector_ref_to_pt_mm = vector_ref_to_pt[::-1].T * np.array(reference_image.GetSpacing())

    # normalise the axis of rotation
    axis_of_rotation = np.array(axis_of_rotation)
    axis_of_rotation = axis_of_rotation / np.linalg.norm(axis_of_rotation)

    # find the normal vectors (i.e. rotation)
    deformation_vectors = np.cross(vector_ref_to_pt_mm, axis_of_rotation[::-1])

    dvf_arr = deformation_vectors * angle * np.pi / 180

    # create a dvf and transform
    dvf_template = sitk.GetImageFromArray(dvf_arr.reshape((*arr.shape, 3)))
    dvf_template.CopyInformation(reference_image)

    dvf_tfm = sitk.DisplacementFieldTransform(sitk.Cast(dvf_template, sitk.sitkVectorFloat64))

    return dvf_template, dvf_tfm


def generate_field_radial_bend(
    reference_image,
    body_mask,
    reference_point,
    axis_of_rotation=[0, 0, -1],
    angle=5.0,
    mask_bend_from_reference_point=("z", "inf"),
    gaussian_smooth=5,
):
    """
    Generates a synthetic field characterised by radial bending.
    Typically, this field would be used to simulate a moving head and so masking is important.

    IMPORTANT! This function relies on the small angle approximation for quick computation.
    This is only valid for angles less than ~20 degrees, so you WILL get strange results
    if you use an angle larger than this (often appears as if regions in the image shrink).

    Args:
        reference_image ([SimpleITK.Image]): The image to be deformed.
        body_mask ([SimpleITK.Image]): A binary mask in which the deformation field will be defined
        reference_point ([tuple]): The point (z,y,x) about which the rotation field is defined.
        axis_of_rotation (tuple, optional): The axis of rotation (z,y,x). Defaults to [0, 0, -1].
        angle (float, optional): The angle of rotation in degrees.
        mask_bend_from_reference_point (tuple, optional): The dimension (z=axial, y=coronal,
                                                          x=sagittal) and limit (inf/sup, post/ant,
                                                          left/right) for masking the vector field,
                                                          relative to the reference point. Defaults
                                                          to ("z", "inf").
        gaussian_smooth (int | list, optional): Scale of a Gaussian kernel used to smooth the
                                                deformation vector field. Defaults to 5.

    Returns:
        [SimpleITK.Image]: The binary mask following the expansion.
        [SimpleITK.DisplacementFieldTransform]: The transform representing the expansion.
        [SimpleITK.Image]: The displacement vector field representing the expansion.
    """

    dvf, tfm_dvf = generate_normal_vectors(
        reference_image=reference_image,
        reference_point=reference_point,
        axis_of_rotation=axis_of_rotation,
        angle=angle,
    )

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

    body_mask = sitk.GetImageFromArray(body_mask_arr)
    body_mask.CopyInformation(reference_image)

    # now we must repeat the process for these new points
    body_mask |= apply_transform(
        body_mask,
        transform=tfm_dvf,
        default_value=0,
        interpolator=sitk.sitkNearestNeighbor,
    )

    dvf = sitk.Mask(dvf, body_mask)

    # smooth
    if np.any(gaussian_smooth):

        if not hasattr(gaussian_smooth, "__iter__"):
            gaussian_smooth = (gaussian_smooth,) * 3

        gaussian_smooth = [
            int(np.ceil(i / j)) for i, j in zip(gaussian_smooth, reference_image.GetSpacing())
        ]

        dvf = sitk.SmoothingRecursiveGaussian(dvf, gaussian_smooth)

    # generate final transform
    tfm_dvf = sitk.DisplacementFieldTransform(sitk.Cast(dvf, sitk.sitkVectorFloat64))

    reference_image_bend = apply_transform(
        reference_image,
        transform=tfm_dvf,
        default_value=int(sitk.GetArrayViewFromImage(reference_image).min()),
        interpolator=sitk.sitkLinear,
    )

    return reference_image_bend, tfm_dvf, dvf, body_mask
