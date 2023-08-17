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

from platipy.imaging.label.fusion import process_probability_image

from platipy.imaging.utils.crop import label_to_roi, crop_to_roi


class PCAFieldModel:
    def __init__(self):
        None


def expansion_model(
    mask,
    mask_bone=None,
    bone_border_mm=4,
    expansion_mm=10,
    gaussian_smooth_mm=None,
    internal_deformation=False,
    initial_dvf=None,
    verbose=False,
):

    if verbose:
        print("Beginning expansion model generation.")

    # crop to ROI
    # use a generous margin
    crop_box_size, crop_box_index = label_to_roi(
        mask, expansion_mm=2 * np.array(expansion_mm) + 25
    )

    mask_crop = crop_to_roi(mask, size=crop_box_size, index=crop_box_index)

    if verbose is True:
        vol_reduction = np.prod(mask.GetSize()) / np.prod(mask_crop.GetSize())
        print(f"Cropped complete. Volume reduction factor: {vol_reduction:.2f}")

    # the DIR process will inherently smooth the DVF
    # so we need to expand a bit more than requested to get to the desired size
    # experiments have shown this to be a factor of approx 30%
    expansion_mm = 1.3 * expansion_mm

    # convert expansion to image units
    expansion_mm = np.array(expansion_mm)
    expansion = expansion_mm / np.array(mask.GetSpacing())
    if verbose:
        print(f"Expansion (voxels): {expansion.astype(int).tolist()}")

    # apply expansion
    # If all negative: erode
    if np.all(np.array(expansion) <= 0):
        print("All factors negative: shrinking only.")
        mask_expand = sitk.BinaryErode(
            mask_crop, np.abs(expansion).astype(int).tolist(), sitk.sitkBall
        )

    # If all positive: dilate
    elif np.all(np.array(expansion) >= 0):
        print("All factors positive: expansion only.")
        mask_expand = sitk.BinaryDilate(
            mask_crop, np.abs(expansion).astype(int).tolist(), sitk.sitkBall
        )

    # Otherwise: sequential operations
    else:
        print("Mixed factors: shrinking and expansion.")
        expansion_kernel = expansion * (expansion > 0)
        shrink_kernel = expansion * (expand < 0)

        mask_expand = sitk.BinaryDilate(
            mask_crop, np.abs(expansion_kernel).astype(int).tolist(), sitk.sitkBall
        )
        mask_expand = sitk.BinaryErode(
            mask_expand, np.abs(shrink_kernel).astype(int).tolist(), sitk.sitkBall
        )

    # if the expansion is too big chances are we won't be able to get there
    # we (somewhat arbitrarily) set this to 10x
    # and if exceeded, simply run at half the requested expansion
    vol_increase = (
        sitk.GetArrayViewFromImage(mask_expand).sum() / sitk.GetArrayViewFromImage(mask_crop).sum()
    )
    if verbose:
        print(f"Volume expansion factor: {vol_increase:.2f}")

    if vol_increase > 50:
        _, _, dvf_inter = expansion_model(
            mask=mask,
            mask_bone=mask_bone,
            expansion_mm=expansion_mm / 2,
            gaussian_smooth_mm=gaussian_smooth_mm,
            internal_deformation=internal_deformation,
            initial_dvf=initial_dvf,
            verbose=verbose,
        )

        # create the initial DVF (if required)
        if initial_dvf is None:
            initial_dvf = sitk.Image(
                mask.GetWidth(),
                mask.GetHeight(),
                mask.GetDepth(),
                sitk.sitkVectorFloat64,
            )
            initial_dvf.CopyInformation(mask)

        # add the intermediate DVF to the initial (which will be zero in most cases)
        tfm_initial = sitk.DisplacementFieldTransform(
            sitk.Cast(initial_dvf, sitk.sitkVectorFloat64)
        )
        initial_dvf = initial_dvf + sitk.Resample(dvf_inter, tfm_initial)

    # apply bone mask if defined
    if mask_bone is not None:
        border = int((bone_border_mm + 0.5) / min(mask.GetSpacing()))

        if verbose:
            print(f"Bone border (voxels) [pre-reg]: {border}")

        mask_bone_border = (np.array(border) / np.array(mask.GetSpacing())).astype(int).tolist()

        mask_bone = crop_to_roi(mask_bone, size=crop_box_size, index=crop_box_index)
        mask_bone_expand = sitk.BinaryDilate(mask_bone, mask_bone_border)

        mask_crop = sitk.MaskNegated(mask_crop, mask_bone_expand)
        mask_expand = sitk.MaskNegated(mask_expand, mask_bone_expand)

    # map to registration structure
    if internal_deformation:
        registration_mask_original = convert_mask_to_reg_structure(mask_crop)
        registration_mask_expand = convert_mask_to_reg_structure(mask_expand)

    else:
        registration_mask_original = mask_crop
        registration_mask_expand = mask_expand

    # we need to find an achievable DVF
    # some heuristics are applied to find appropriate settings
    # first the scale of the regularisation kernel
    # some work needed here
    # for example to take into account the size of the structure
    max_expansion = expansion_mm.max()
    regularisation_kernel_mm = 1.5  # 0.75 * np.exp(0.06 * max_expansion) - 0.5

    # second the resolution staging
    min_resolution = min([min(mask.GetSpacing()), max_expansion.min()])
    resolution_staging = [min_resolution * 2 ** i for i in range(3)][::-1]

    if verbose is True:
        print(f"Regularisation kernel (mm): {regularisation_kernel_mm:.2f}")
        print(f"Resolution staging: {resolution_staging}")

    _, _, dvf_crop = fast_symmetric_forces_demons_registration(
        registration_mask_expand,
        registration_mask_original,
        isotropic_resample=True,
        resolution_staging=resolution_staging,
        iteration_staging=[125, 80, 50],
        regularisation_kernel_mm=regularisation_kernel_mm,
        smoothing_sigma_factor=0,
        initial_displacement_field=initial_dvf,
        ncores=8,
        verbose=False,
    )

    # smooth is required
    if gaussian_smooth_mm is not None:
        gaussian_smooth = np.array(gaussian_smooth_mm) / np.array(mask.GetSpacing())

        if verbose:
            print(f"Gaussian smoothing kernel size (voxels): {gaussian_smooth}")

        dvf_crop = sitk.SmoothingRecursiveGaussian(dvf_crop, gaussian_smooth.astype(int).tolist())

    # clean up vector field
    # we define a border equal to 4 mm
    if mask_bone is not None:
        border = int((bone_border_mm + 0.5) / min(mask.GetSpacing()))

        if verbose:
            print(f"Bone border (voxels) [post-reg]: {border}")

        mask_bone_border = (np.array(border) / np.array(mask.GetSpacing())).astype(int).tolist()

        dvf_crop = sitk.Mask(dvf_crop, sitk.Not(sitk.BinaryDilate(mask_bone, mask_bone_border)))
        dvf_crop = sitk.SmoothingRecursiveGaussian(dvf_crop, mask_bone_border)

    # uncrop vector field
    dvf_empty = sitk.Image(mask.GetSize(), sitk.sitkVectorFloat64, 3)
    dvf_empty.CopyInformation(mask)

    dvf = sitk.Paste(
        dvf_empty,
        sitk.Cast(dvf_crop, sitk.sitkVectorFloat64),
        dvf_crop.GetSize(),
        (0, 0, 0),
        crop_box_index,
    )

    # define the transformation
    tfm = sitk.DisplacementFieldTransform(sitk.Cast(dvf, sitk.sitkVectorFloat64))

    # apply transformation
    mask_expand = apply_transform(
        sitk.Cast(mask, sitk.sitkFloat32),
        transform=tfm,
        default_value=0,
        interpolator=sitk.sitkLinear,
    )

    mask_expand = process_probability_image(mask_expand)

    return mask_expand, tfm, dvf
