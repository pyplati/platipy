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
    smooth_and_resample,
    apply_transform,
    registration_command_iteration,
    stage_iteration,
    deformable_registration_command_iteration,
    control_point_spacing_distance_to_number,
)


def multiscale_demons(
    registration_algorithm,
    fixed_image,
    moving_image,
    initial_transform=None,
    initial_displacement_field=None,
    isotropic_resample=None,
    resolution_staging=None,
    smoothing_sigmas=None,
    iteration_staging=None,
):
    """
    Run the given registration algorithm in a multiscale fashion. The original scale should not be
    given as input as the original images are implicitly incorporated as the base of the pyramid.
    Args:
        registration_algorithm: Any registration algorithm that has an Execute(fixed_image,
                                moving_image, displacement_field_image) method.
        fixed_image: Resulting transformation maps points from this image's spatial domain to the
                     moving image spatial domain.
        moving_image: Resulting transformation maps points from the fixed_image's spatial domain to
                      this image's spatial domain.
        initial_transform: Any SimpleITK transform, used to initialize the displacement field.
        initial_displacement_field: Initial displacement field, if this is provided
                                    initial_transform will be ignored
        shrink_factors: Shrink factors relative to the original image's size.
        smoothing_sigmas: Amount of smoothing which is done prior to resmapling the image using the
                          given shrink factor. These are in physical (image spacing) units.
    Returns:
        SimpleITK.DisplacementFieldTransform
        [Optional] Displacemment (vector) field
    """
    # Create image pyramid.
    fixed_images = []
    moving_images = []

    for resolution, smoothing_sigma in reversed(list(zip(resolution_staging, smoothing_sigmas))):

        isotropic_voxel_size_mm = None
        shrink_factor = None

        if isotropic_resample:
            isotropic_voxel_size_mm = resolution
        else:
            shrink_factor = resolution

        fixed_images.append(
            smooth_and_resample(
                fixed_image,
                isotropic_voxel_size_mm=isotropic_voxel_size_mm,
                shrink_factor=shrink_factor,
                smoothing_sigma=smoothing_sigma,
            )
        )
        moving_images.append(
            smooth_and_resample(
                moving_image,
                isotropic_voxel_size_mm=isotropic_voxel_size_mm,
                shrink_factor=shrink_factor,
                smoothing_sigma=smoothing_sigma,
            )
        )

    # Create initial displacement field at lowest resolution.
    # Currently, the pixel type is required to be sitkVectorFloat64 because of a constraint imposed
    # by the Demons filters.
    if not initial_displacement_field:
        if initial_transform:
            initial_displacement_field = sitk.TransformToDisplacementField(
                initial_transform,
                sitk.sitkVectorFloat64,
                fixed_images[-1].GetSize(),
                fixed_images[-1].GetOrigin(),
                fixed_images[-1].GetSpacing(),
                fixed_images[-1].GetDirection(),
            )
        else:
            if len(moving_image.GetSize()) == 2:
                initial_displacement_field = sitk.Image(
                    fixed_images[-1].GetWidth(),
                    fixed_images[-1].GetHeight(),
                    sitk.sitkVectorFloat64,
                )
            elif len(moving_image.GetSize()) == 3:
                initial_displacement_field = sitk.Image(
                    fixed_images[-1].GetWidth(),
                    fixed_images[-1].GetHeight(),
                    fixed_images[-1].GetDepth(),
                    sitk.sitkVectorFloat64,
                )
            initial_displacement_field.CopyInformation(fixed_images[-1])
    else:
        initial_displacement_field = sitk.Resample(initial_displacement_field, fixed_images[-1])

    # Run the registration.
    iters = iteration_staging[0]
    registration_algorithm.SetNumberOfIterations(iters)
    initial_displacement_field = registration_algorithm.Execute(
        fixed_images[-1], moving_images[-1], initial_displacement_field
    )
    # Start at the top of the pyramid and work our way down.
    for i, (f_image, m_image) in enumerate(
        reversed(list(zip(fixed_images[0:-1], moving_images[0:-1])))
    ):
        initial_displacement_field = sitk.Resample(initial_displacement_field, f_image)
        iters = iteration_staging[i + 1]
        registration_algorithm.SetNumberOfIterations(iters)
        initial_displacement_field = registration_algorithm.Execute(
            f_image, m_image, initial_displacement_field
        )

    output_displacement_field = sitk.Resample(
        initial_displacement_field, initial_displacement_field
    )

    return sitk.DisplacementFieldTransform(initial_displacement_field), output_displacement_field


def fast_symmetric_forces_demons_registration(
    fixed_image,
    moving_image,
    resolution_staging=[8, 4, 1],
    iteration_staging=[10, 10, 10],
    isotropic_resample=False,
    initial_displacement_field=None,
    smoothing_sigma_factor=1,
    smoothing_sigmas=False,
    default_value=-1000,
    ncores=1,
    interp_order=2,
    verbose=False,
):
    """
    Deformable image propagation using Fast Symmetric-Forces Demons

    Args
        fixed_image (sitk.Image)        : the fixed image
        moving_image (sitk.Image)       : the moving image, to be deformable registered (must be in
                                          the same image space)
        resolution_staging (list[int])   : down-sampling factor for each resolution level
        iteration_staging (list[int])    : number of iterations for each resolution level
        isotropic_resample (bool)        : flag to request isotropic resampling of images, in which
                                           case resolution_staging is used to define voxel size
                                           (mm) per level
        initial_displacement_field (sitk.Image) : Initial displacement field to use
        ncores (int)                    : number of processing cores to use
        smoothing_sigma_factor (float)    : the relative width of the Gaussian smoothing kernel
        interp_order (int)               : the interpolation order
                                            1 = Nearest neighbour
                                            2 = Bi-linear splines
                                            3 = B-Spline (cubic)

    Returns
        registered_image (sitk.Image)    : the registered moving image
        output_transform                 : the displacement field transform
        [optional] deformation_field
    """

    # Cast to floating point representation, if necessary

    moving_image_type = moving_image.GetPixelID()

    if fixed_image.GetPixelID() != 6:
        fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    if moving_image.GetPixelID() != 6:
        moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    # Set up the appropriate image filter
    registration_method = sitk.FastSymmetricForcesDemonsRegistrationFilter()

    # Multi-resolution framework
    registration_method.SetNumberOfThreads(ncores)
    registration_method.SetSmoothUpdateField(True)
    registration_method.SetSmoothDisplacementField(True)
    registration_method.SetStandardDeviations(1.5)

    # This allows monitoring of the progress
    if verbose:
        registration_method.AddCommand(
            sitk.sitkIterationEvent,
            lambda: deformable_registration_command_iteration(registration_method),
        )

    if not smoothing_sigmas:
        smoothing_sigmas = [i * smoothing_sigma_factor for i in resolution_staging]

    output_transform, deformation_field = multiscale_demons(
        registration_algorithm=registration_method,
        fixed_image=fixed_image,
        moving_image=moving_image,
        resolution_staging=resolution_staging,
        smoothing_sigmas=smoothing_sigmas,
        iteration_staging=iteration_staging,
        isotropic_resample=isotropic_resample,
        initial_displacement_field=initial_displacement_field,
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(interp_order)
    resampler.SetDefaultPixelValue(default_value)

    resampler.SetTransform(output_transform)
    registered_image = resampler.Execute(moving_image)

    registered_image.CopyInformation(fixed_image)
    registered_image = sitk.Cast(registered_image, moving_image_type)

    resampled_field = sitk.Resample(deformation_field, fixed_image)

    return registered_image, output_transform, resampled_field


def bspline_registration(
    fixed_image,
    moving_image,
    fixed_structure=False,
    moving_structure=False,
    resolution_staging=[8, 4, 2],
    smooth_sigmas=[4, 2, 1],
    sampling_rate=0.1,
    optimiser="LBFGS",
    metric="mean_squares",
    initial_grid_spacing=64,
    grid_scale_factors=[1, 2, 4],
    interp_order=3,
    default_value=-1000,
    number_of_iterations=20,
    isotropic_resample=False,
    initial_isotropic_size=1,
    number_of_histogram_bins_mi=30,
    verbose=False,
    ncores=8,
):
    """
    B-Spline image registration using ITK

    IMPORTANT - THIS IS UNDER ACTIVE DEVELOPMENT

    Args:
        fixed_image ([SimpleITK.Image]): The fixed (target/primary) image.
        moving_image ([SimpleITK.Image]): The moving (secondary) image.
        fixed_structure (bool, optional): If defined, a binary SimpleITK.Image used to mask metric
                                          evaluation for the moving image. Defaults to False.
        moving_structure (bool, optional): If defined, a binary SimpleITK.Image used to mask metric
                                           evaluation for the fixed image. Defaults to False.
        resolution_staging (list, optional): The multi-resolution downsampling factors.
                                             Defaults to [8, 4, 2].
        smooth_sigmas (list, optional): The multi-resolution smoothing kernel scale (Gaussian).
                                        Defaults to [4, 2, 1].
        sampling_rate (float, optional): The fraction of voxels sampled during each iteration.
                                         Defaults to 0.1.
        optimiser (str, optional): The optimiser algorithm used for image registration.
                                   Available options:
                                    - LBFSGS
                                      (limited-memory Broyden–Fletcher–Goldfarb–Shanno (bounded).)
                                    - LBFSG
                                      (limited-memory Broyden–Fletcher–Goldfarb–Shanno
                                      (unbounded).)
                                    - CGLS (conjugate gradient line search)
                                    - gradient_descent
                                    - gradient_descent_line_search
                                   Defaults to "LBFGS".
        metric (str, optional): The metric to be optimised during image registration.
                                Available options:
                                 - correlation
                                 - mean_squares
                                 - demons
                                 - mutual_information
                                   (used with parameter number_of_histogram_bins_mi)
                                Defaults to "mean_squares".
        initial_grid_spacing (int, optional): Grid spacing of lower resolution stage (in mm).
                                              Defaults to 64.
        grid_scale_factors (list, optional): Factors to determine grid spacing at each
                                             multiresolution stage.
                                             Defaults to [1, 2, 4].
        interp_order (int, optional): Interpolation order of final resampling.
                                      Defaults to 3 (cubic).
        default_value (int, optional): Default image value. Defaults to -1000.
        number_of_iterations (int, optional): Number of iterations at each resolution stage.
                                              Defaults to 20.
        isotropic_resample (bool, optional): Flag whether to resample to isotropic resampling
                                             prior to registration.
                                             Defaults to False.
        initial_isotropic_size (int, optional): Voxel size (in mm) of resampled isotropic image
                                                (if used). Defaults to 1.
        number_of_histogram_bins_mi (int, optional): Number of histogram bins used when calculating
                                                     mutual information. Defaults to 30.
        verbose (bool, optional): Print image registration process information. Defaults to False.
        ncores (int, optional): Number of CPU cores used. Defaults to 8.

    Returns:
        [SimpleITK.Image]: The registered moving (secondary) image.
        [SimleITK.Transform]: The linear transformation.

    Notes:
     - smooth_sigmas are relative to resolution staging
        e.g. for image spacing of 1x1x1 mm^3, with smooth sigma=2 and resolution_staging=4, the
        scale of the Gaussian filter would be 2x4 = 8mm (i.e. 8x8x8 mm^3)

    """

    # Re-cast input images
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)

    moving_image_type = moving_image.GetPixelID()
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    # (Optional) isotropic resample
    # This changes the behaviour, so care should be taken
    # For highly anisotropic images may be preferable

    if isotropic_resample:
        # First, copy the fixed image so we can resample back into this space at the end
        fixed_image_original = fixed_image
        fixed_image_original.MakeUnique()

        fixed_image = smooth_and_resample(
            fixed_image,
            isotropic_voxel_size_mm=initial_isotropic_size,
        )

        moving_image = smooth_and_resample(
            moving_image,
            isotropic_voxel_size_mm=initial_isotropic_size,
        )

    else:
        fixed_image_original = fixed_image

    # Set up image registration method
    registration = sitk.ImageRegistrationMethod()
    registration.SetNumberOfThreads(ncores)

    registration.SetShrinkFactorsPerLevel(resolution_staging)
    registration.SetSmoothingSigmasPerLevel(smooth_sigmas)
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Choose optimiser
    if optimiser.lower() == "lbfgsb":
        registration.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=number_of_iterations,
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=1024,
            costFunctionConvergenceFactor=1e7,
            trace=verbose,
        )
    elif optimiser.lower() == "lbfgs":
        registration.SetOptimizerAsLBFGS2(
            numberOfIterations=number_of_iterations,
            solutionAccuracy=1e-2,
            hessianApproximateAccuracy=6,
            deltaConvergenceDistance=0,
            deltaConvergenceTolerance=0.01,
            lineSearchMaximumEvaluations=40,
            lineSearchMinimumStep=1e-20,
            lineSearchMaximumStep=1e20,
            lineSearchAccuracy=0.01,
        )
    elif optimiser.lower() == "cgls":
        registration.SetOptimizerAsConjugateGradientLineSearch(
            learningRate=0.05, numberOfIterations=number_of_iterations
        )
        registration.SetOptimizerScalesFromPhysicalShift()
    elif optimiser.lower() == "gradient_descent":
        registration.SetOptimizerAsGradientDescent(
            learningRate=5.0,
            numberOfIterations=number_of_iterations,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
        registration.SetOptimizerScalesFromPhysicalShift()
    elif optimiser.lower() == "gradient_descent_line_search":
        registration.SetOptimizerAsGradientDescentLineSearch(
            learningRate=1.0, numberOfIterations=number_of_iterations
        )
        registration.SetOptimizerScalesFromPhysicalShift()

    # Set metric
    if metric == "correlation":
        registration.SetMetricAsCorrelation()
    elif metric == "mean_squares":
        registration.SetMetricAsMeanSquares()
    elif metric == "demons":
        registration.SetMetricAsDemons()
    elif metric == "mutual_information":
        registration.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=number_of_histogram_bins_mi
        )

    registration.SetInterpolator(sitk.sitkLinear)

    # Set sampling
    if isinstance(sampling_rate, float):
        registration.SetMetricSamplingPercentage(sampling_rate)
    elif type(sampling_rate) in [np.ndarray, list]:
        registration.SetMetricSamplingPercentagePerLevel(sampling_rate)

    registration.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.REGULAR)

    # Set masks
    if moving_structure is not False:
        registration.SetMetricMovingMask(moving_structure)

    if fixed_structure is not False:
        registration.SetMetricFixedMask(fixed_structure)

    # Set control point spacing
    transform_domain_mesh_size = control_point_spacing_distance_to_number(
        fixed_image, initial_grid_spacing
    )

    if verbose:
        print(f"Initial grid size: {transform_domain_mesh_size}")

    # Initialise transform
    initial_transform = sitk.BSplineTransformInitializer(
        fixed_image,
        transformDomainMeshSize=[int(i) for i in transform_domain_mesh_size],
    )
    registration.SetInitialTransformAsBSpline(
        initial_transform, inPlace=True, scaleFactors=grid_scale_factors
    )

    # (Optionally) add iteration commands
    if verbose:
        registration.AddCommand(
            sitk.sitkIterationEvent,
            lambda: registration_command_iteration(registration),
        )
        registration.AddCommand(
            sitk.sitkMultiResolutionIterationEvent,
            lambda: stage_iteration(registration),
        )

    # Run the registration
    output_transform = registration.Execute(fixed=fixed_image, moving=moving_image)

    # Resample moving image
    registered_image = apply_transform(
        input_image=moving_image,
        reference_image=fixed_image_original,
        transform=output_transform,
        default_value=default_value,
        interpolator=interp_order,
    )

    registered_image = sitk.Cast(registered_image, moving_image_type)

    # Return outputs
    return registered_image, output_transform
