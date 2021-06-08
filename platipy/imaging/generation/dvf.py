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
)

from platipy.imaging.registration.deformable import (
    fast_symmetric_forces_demons_registration,
)


def generate_field_shift(mask_image, vector_shift=(10, 10, 10), gaussian_smooth=5):
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

    dvf_tfm = sitk.DisplacementFieldTransform(sitk.Cast(dvf_template, sitk.sitkVectorFloat64))
    mask_image_shift = apply_transform(
        mask_image, transform=dvf_tfm, default_value=0, interpolator=sitk.sitkNearestNeighbor
    )

    dvf_template = sitk.Mask(dvf_template, mask_image | mask_image_shift)

    # smooth
    if np.any(gaussian_smooth):

        if not hasattr(gaussian_smooth, "__iter__"):
            gaussian_smooth = (gaussian_smooth,) * 3

        dvf_template = sitk.SmoothingRecursiveGaussian(dvf_template, gaussian_smooth)

    dvf_tfm = sitk.DisplacementFieldTransform(sitk.Cast(dvf_template, sitk.sitkVectorFloat64))
    mask_image_shift = apply_transform(
        mask_image, transform=dvf_tfm, default_value=0, interpolator=sitk.sitkNearestNeighbor
    )

    return mask_image_shift, dvf_tfm, dvf_template


def generate_field_asymmetric_contract(
    mask_image, vector_asymmetric_contract=(10, 10, 10), gaussian_smooth=5
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

    dvf_tfm = sitk.DisplacementFieldTransform(sitk.Cast(dvf_template, sitk.sitkVectorFloat64))

    mask_image_asymmetric_contract = apply_transform(
        mask_image, transform=dvf_tfm, default_value=0, interpolator=sitk.sitkNearestNeighbor
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
    expand=3,
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

    if bone_mask is not False:
        mask_original = mask + bone_mask
    else:
        mask_original = mask

    # Use binary erosion to create a smaller volume
    if not hasattr(expand, "__iter__"):
        expand = (expand,) * 3

    expand = np.array(expand)

    # Convert voxels to millimetres
    expand = expand / np.array(mask.GetSpacing()[::-1])

    # Re-order to (x,y,z)
    expand = expand[::-1]
    # expand = [int(i / j) for i, j in zip(expand, mask.GetSpacing()[::-1])][::-1]

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
        iteration_staging=[10, 10],
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
        scale (int, optional): The deformation vector length at each point will equal scale
                              multiplied by the distance to that point from reference_point.
                              Defaults to 1.
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

    dvf_tfm = sitk.DisplacementFieldTransform(sitk.Cast(dvf_template, sitk.sitkVectorFloat64))
    reference_image_bend = apply_transform(
        reference_image,
        transform=dvf_tfm,
        default_value=int(sitk.GetArrayViewFromImage(reference_image).min()),
        interpolator=sitk.sitkLinear,
    )

    return reference_image_bend, dvf_tfm, dvf_template
