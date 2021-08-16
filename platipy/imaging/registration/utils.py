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

from loguru import logger


def registration_command_iteration(method):
    """
    Utility function to print information during (rigid, similarity, translation, B-splines)
    registration
    """
    print("{0:3} = {1:10.5f}".format(method.GetOptimizerIteration(), method.GetMetricValue()))


def stage_iteration(method):
    """
    Utility function to print information during stage change in registration
    """
    print(f"Number of parameters = {method.GetInitialTransform().GetNumberOfParameters()}")


def deformable_registration_command_iteration(method):
    """
    Utility function to print information during demons registration
    """
    print("{0:3} = {1:10.5f}".format(method.GetElapsedIterations(), method.GetMetric()))


def control_point_spacing_distance_to_number(image, grid_spacing):
    """
    Convert grid spacing specified in distance to number of control points
    """
    image_spacing = np.array(image.GetSpacing())
    image_size = np.array(image.GetSize())
    number_points = image_size * image_spacing / np.array(grid_spacing)
    return (number_points + 0.5).astype(int)


def apply_linear_transform(
    input_image,
    reference_image,
    transform,
    is_structure=False,
    default_value=0,
    interpolator=sitk.sitkNearestNeighbor,
):
    """
    Helper function for applying linear transforms.

    Args
        input_image (SimpleITK.Image): The image, to which the transform is applied
        reference_image (SimpleITK.Image): The image will be resampled into this reference space.
        transform (SimpleITK.Transform): The transformation
        is_structure (bool): Will set appropriate parameters for transforming labels/structures.
        default_value: Default (background) value. Defaults to 0.
        interpolator (int, optional): The interpolation order.
                                Available options:
                                    - SimpleITK.sitkNearestNeighbor
                                    - SimpleITK.sitkLinear
                                    - SimpleITK.sitkBSpline
                                Defaults to SimpleITK.sitkNearestNeighbor

    Returns
        (SimpleITK.Image): the transformed image
    """
    if is_structure:

        if default_value != 0 or interpolator != sitk.sitkNearestNeighbor:
            logger.warning(
                "is_structure is set to True, but you have set default_value "
                "and/or interpolator. default_value and/or interpolator will be overwritten."
            )

        default_value = 0
        interpolator = sitk.sitkNearestNeighbor

    return apply_transform(
        input_image=input_image,
        reference_image=reference_image,
        transform=transform,
        default_value=default_value,
        interpolator=interpolator,
    )


def apply_deformable_transform(
    input_image,
    transform,
    is_structure=False,
    default_value=0,
    interpolator=sitk.sitkNearestNeighbor,
):
    """
    Helper function for applying deformable transforms.

    Args
        input_image (SimpleITK.Image): The image, to which the transform is applied
        reference_image (SimpleITK.Image): The image will be resampled into this reference space.
        transform (SimpleITK.Transform): The transformation
        is_structure (bool): Will set appropriate parameters for transforming labels/structures.
        default_value: Default (background) value. Defaults to 0.
        interpolator (int, optional): The interpolation order.
                                Available options:
                                    - SimpleITK.sitkNearestNeighbor
                                    - SimpleITK.sitkLinear
                                    - SimpleITK.sitkBSpline
                                Defaults to SimpleITK.sitkNearestNeighbor

    Returns
        (SimpleITK.Image): the transformed image
    """

    if is_structure:

        if default_value != 0 or interpolator != sitk.sitkNearestNeighbor:
            logger.warning(
                "is_structure is set to True, but you have set default_value "
                "and/or interpolator. default_value and/or interpolator will be overwritten."
            )

        default_value = 0
        interpolator = sitk.sitkNearestNeighbor

    return apply_transform(
        input_image=input_image,
        reference_image=None,
        transform=transform,
        default_value=default_value,
        interpolator=interpolator,
    )


def apply_transform(
    input_image,
    reference_image=None,
    transform=None,
    default_value=0,
    interpolator=sitk.sitkNearestNeighbor,
):
    """
    Transform a volume of structure with the given deformation field.

    Args
        input_image (SimpleITK.Image): The image, to which the transform is applied
        reference_image (SimpleITK.Image): The image will be resampled into this reference space.
        transform (SimpleITK.Transform): The transformation
        default_value: Default (background) value. Defaults to 0.
        interpolator (int, optional): The interpolation order.
                                Available options:
                                    - SimpleITK.sitkNearestNeighbor
                                    - SimpleITK.sitkLinear
                                    - SimpleITK.sitkBSpline
                                Defaults to SimpleITK.sitkNearestNeighbor

    Returns
        (SimpleITK.Image): the transformed image

    """
    original_image_type = input_image.GetPixelID()

    resampler = sitk.ResampleImageFilter()

    if reference_image:
        resampler.SetReferenceImage(reference_image)
    else:
        resampler.SetReferenceImage(input_image)

    if transform:
        resampler.SetTransform(transform)

    resampler.SetDefaultPixelValue(default_value)
    resampler.SetInterpolator(interpolator)

    output_image = resampler.Execute(input_image)
    output_image = sitk.Cast(output_image, original_image_type)

    return output_image


def smooth_and_resample(
    image,
    isotropic_voxel_size_mm=None,
    shrink_factor=None,
    smoothing_sigma=None,
    interpolator=sitk.sitkLinear,
):
    """
    Args:
        image (SimpleITK.Image): The image we want to resample.
        isotropic_voxel_size_mm (float | None): New voxel size in millimetres
        shrink_factor (list | float): A number greater than one, such that the new image's size is
            original_size/shrink_factor. Can also be specified independently for each
            dimension (sagittal, coronal, axial).
        smoothing_sigma (list | float): Scale for Gaussian smoothing, this is in physical
            (image spacing) units, not pixels. Can also be specified independently for
            each dimension (sagittal, coronal, axial).
    Return:
        SimpleITK.Image: Image which is a result of smoothing the input and then resampling
        it using the specified Gaussian kernel and shrink factor.
    """
    if smoothing_sigma:
        if hasattr(smoothing_sigma, "__iter__"):
            smoothing_variance = [i * i for i in smoothing_sigma]
        else:
            smoothing_variance = (smoothing_sigma ** 2,) * 3

        maximum_kernel_width = int(
            max([8 * j * i for i, j in zip(image.GetSpacing(), smoothing_variance)])
        )

        image = sitk.DiscreteGaussian(image, smoothing_variance, maximum_kernel_width)

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    if shrink_factor and isotropic_voxel_size_mm:
        raise AttributeError(
            "Function must be called with either isotropic_voxel_size_mm or "
            "shrink_factor, not both."
        )

    elif isotropic_voxel_size_mm:
        scale_factor = (
            isotropic_voxel_size_mm * np.ones_like(image.GetSize()) / np.array(image.GetSpacing())
        )
        new_size = [int(sz / float(sf) + 0.5) for sz, sf in zip(original_size, scale_factor)]

    elif shrink_factor:
        if isinstance(shrink_factor, list):
            new_size = [int(sz / float(sf) + 0.5) for sz, sf in zip(original_size, shrink_factor)]
        else:
            new_size = [int(sz / float(shrink_factor) + 0.5) for sz in original_size]

    else:
        return image

    new_spacing = [
        ((size_o_i - 1) * spacing_o_i) / (size_n_i - 1)
        for size_o_i, spacing_o_i, size_n_i in zip(original_size, original_spacing, new_size)
    ]

    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        interpolator,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )


def convert_mask_to_distance_map(mask, squared_distance=False, normalise=False):
    """
    Generate a distance map from a binary label.

    Args:
        mask ([SimpleITK.Image]): A binary label.
        squared_distance (bool, optional): Option to use the squared distance. Defaults to False.
        normalise (bool, optional): Normalise output to [0,1]. Defaults to False.

    Returns:
        [SimpleITK.Image]: The distance map as an image.
    """
    arr = sitk.GetArrayViewFromImage(mask)
    vals = np.unique(arr[arr > 0])
    if len(vals) > 2:
        # There is more than one value! We need to threshold at the median
        cutoff = np.median(vals)
        mask = sitk.BinaryThreshold(mask, cutoff, np.max(vals).astype(float))

    raw_map = sitk.SignedMaurerDistanceMap(
        mask,
        insideIsPositive=True,
        squaredDistance=squared_distance,
        useImageSpacing=True,
    )

    if normalise:
        return raw_map / (sitk.GetArrayFromImage(raw_map).max())
    else:
        return raw_map


def convert_mask_to_reg_structure(mask, expansion=(0, 0, 0), scale=lambda x: x):
    """
    Generate a mask-like image to make structure-guided registration more
    realistic via internal deformation within a binary mask.

    Args:
        mask ([SimpleITK.Image]): The binary label.
        expansion (int, optional): For improved smoothness on the surface
                                   (particularly complex structures) it will
                                   often help to use some binary dilation. This
                                   parameter defines the expansion in mm.
                                   Defaults to 1.
        scale ([function], optional): Defines scaling to the distance map.
                                      For example: lambda x:sitk.Log(x).
                                      Defaults to lambda x:x.

    Returns:
        [SimpleITK.Image]: [description]
    """
    arr = sitk.GetArrayViewFromImage(mask)
    vals = np.unique(arr[arr > 0])
    if len(vals) > 2:
        # There is more than one value! We need to threshold at the median
        cutoff = np.median(vals)
        mask = sitk.BinaryThreshold(mask, cutoff, np.max(vals).astype(float))

    if not hasattr(expansion, "__iter__"):
        expansion = [int(expansion / i) for i in mask.GetSpacing()]
    if any(expansion):
        mask = sitk.BinaryDilate(mask, expansion)

    distance_map = sitk.Cast(
        convert_mask_to_distance_map(mask, squared_distance=False), sitk.sitkFloat64
    )

    distance_map = sitk.Mask(
        distance_map,
        mask,
    )

    scaled_distance_map = distance_map / (sitk.GetArrayViewFromImage(distance_map).max())

    return scale(scaled_distance_map)
