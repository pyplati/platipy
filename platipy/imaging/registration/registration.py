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


import sys
import numpy as np
import SimpleITK as sitk


def convert_mask_to_distance_map(mask, squaredDistance=False, normalise=False):
    raw_map = sitk.SignedMaurerDistanceMap(
        mask,
        insideIsPositive=True,
        squaredDistance=squaredDistance,
        useImageSpacing=True,
    )

    if normalise:
        return raw_map / (sitk.GetArrayFromImage(raw_map).max())
    else:
        return raw_map


def convert_mask_to_reg_structure(mask, expansion=1, scale=lambda x: x):
    distance_map = sitk.Cast(
        convert_mask_to_distance_map(mask, squaredDistance=False), sitk.sitkFloat64
    )

    inverted_distance_map = sitk.Threshold(
        distance_map
        + expansion * sitk.Cast(distance_map < (expansion), sitk.sitkFloat64),
        lower=0,
        upper=1000,
    )

    scaled_distance_map = inverted_distance_map / (
        sitk.GetArrayViewFromImage(inverted_distance_map).max()
    )

    return scale(scaled_distance_map)


def initial_registration_command_iteration(method):
    """
    Utility function to print information during initial (rigid, similarity, affine, translation) registration
    """
    print(
        "{0:3} = {1:10.5f}".format(
            method.GetOptimizerIteration(), method.GetMetricValue()
        )
    )


def deformable_registration_command_iteration(method):
    """
    Utility function to print information during demons registration
    """
    print("{0:3} = {1:10.5f}".format(method.GetElapsedIterations(), method.GetMetric()))


def stage_iteration(method):
    """
    Utility function to print information during stage change in registration
    """
    print(
        f"Number of parameters = {method.GetInitialTransform().GetNumberOfParameters()}"
    )


def control_point_spacing_distance_to_number(image, grid_spacing):
    """
    Convert grid spacing specified in distance to number of control points
    """
    image_spacing = np.array(image.GetSpacing())
    image_size = np.array(image.GetSize())
    number_points = image_size * image_spacing / np.array(grid_spacing)
    return (number_points + 0.5).astype(int)


def alignment_registration(fixed_image, moving_image, default_value=0, moments=True):
    moving_image_type = moving_image.GetPixelIDValue()
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image, sitk.VersorRigid3DTransform(), moments
    )
    aligned_image = sitk.Resample(moving_image, fixed_image, initial_transform)
    aligned_image = sitk.Cast(aligned_image, moving_image_type)
    return aligned_image, initial_transform


def initial_registration(
    fixed_image,
    moving_image,
    moving_structure=False,
    fixed_structure=False,
    options={
        "shrink_factors": [8, 2, 1],
        "smooth_sigmas": [4, 2, 0],
        "sampling_rate": 0.1,
        "final_interp": 3,
        "metric": "mean_squares",
        "optimiser": "gradient_descent",
        "number_of_iterations": 50,
    },
    default_value=-1024,
    trace=False,
    reg_method="Similarity",
):
    """
    Rigid image registration using ITK

    Args
        fixed_image (sitk.Image) : the fixed image
        moving_image (sitk.Image): the moving image, transformed to match fixed_image
        options (dict)          : registration options
        structure (bool)        : True if the image is a structure image

    Returns
        registered_image (sitk.Image): the rigidly registered moving image
        transform (transform        : the transform, can be used directly with
                                      sitk.ResampleImageFilter

    """

    # Re-cast
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)

    moving_image_type = moving_image.GetPixelIDValue()
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    # Get the options
    shrink_factors = options["shrink_factors"]
    smooth_sigmas = options["smooth_sigmas"]
    sampling_rate = options["sampling_rate"]
    final_interp = options["final_interp"]
    metric = options["metric"]
    optimiser = options["optimiser"]
    number_of_iterations = options["number_of_iterations"]

    # Initialise using a VersorRigid3DTransform
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image, sitk.Euler3DTransform(), False
    )
    # Set up image registration method
    registration = sitk.ImageRegistrationMethod()

    registration.SetShrinkFactorsPerLevel(shrink_factors)
    registration.SetSmoothingSigmasPerLevel(smooth_sigmas)
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration.SetMovingInitialTransform(initial_transform)

    if metric == "correlation":
        registration.SetMetricAsCorrelation()
    elif metric == "mean_squares":
        registration.SetMetricAsMeanSquares()
    elif metric == "mattes_mi":
        registration.SetMetricAsMattesMutualInformation()
    elif metric == "joint_hist_mi":
        registration.SetMetricAsJointHistogramMutualInformation()
    elif metric == "ants":
        try:
            ants_radius = options["ants_radius"]
        except:
            ants_radius = 3
        registration.SetMetricAsANTSNeighborhoodCorrelation(ants_radius)
    # to do: add the rest

    registration.SetInterpolator(sitk.sitkLinear)  # Perhaps a small gain in improvement
    registration.SetMetricSamplingPercentage(sampling_rate)
    registration.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.REGULAR)

    # This is only necessary if using a transform comprising changes with different units
    # e.g. rigid (rotation: radians, translation: mm)
    # It can safely be left on
    registration.SetOptimizerScalesFromPhysicalShift()

    if moving_structure:
        registration.SetMetricMovingMask(moving_structure)

    if fixed_structure:
        registration.SetMetricFixedMask(fixed_structure)

    if reg_method.lower() == "translation":
        registration.SetInitialTransform(sitk.TranslationTransform(3))
    elif reg_method.lower() == "similarity":
        registration.SetInitialTransform(sitk.Similarity3DTransform())
    elif reg_method.lower() == "affine":
        registration.SetInitialTransform(sitk.AffineTransform(3))
    elif reg_method.lower() == "rigid":
        registration.SetInitialTransform(sitk.VersorRigid3DTransform())
    elif reg_method.lower() == "scaleversor":
        registration.SetInitialTransform(sitk.ScaleVersor3DTransform())
    elif reg_method.lower() == "scaleskewversor":
        registration.SetInitialTransform(sitk.ScaleSkewVersor3DTransform())
    else:
        raise ValueError(
            "You have selected a registration method that does not exist.\n Please select from Translation, Similarity, Affine, Rigid"
        )

    if optimiser.lower() == "lbfgsb":
        registration.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=number_of_iterations,
            maximumNumberOfCorrections=50,
            maximumNumberOfFunctionEvaluations=1024,
            costFunctionConvergenceFactor=1e7,
            trace=trace,
        )
    elif optimiser.lower() == "exhaustive":
        """
        This isn't well implemented
        Needs some work to give options for sampling rates
        Use is not currently recommended
        """
        samples = [10, 10, 10, 10, 10, 10]
        registration.SetOptimizerAsExhaustive(samples)
    elif optimiser.lower() == "gradient_descent_line_search":
        registration.SetOptimizerAsGradientDescentLineSearch(
            learningRate=1.0, numberOfIterations=number_of_iterations
        )
    elif optimiser.lower() == "gradient_descent":
        registration.SetOptimizerAsGradientDescent(
            learningRate=1.0, numberOfIterations=number_of_iterations
        )

    if trace:
        registration.AddCommand(
            sitk.sitkIterationEvent,
            lambda: initial_registration_command_iteration(registration),
        )

    output_transform = registration.Execute(fixed=fixed_image, moving=moving_image)
    # Combine initial and optimised transform
    combined_transform = sitk.CompositeTransform([initial_transform, output_transform])

    registered_image = transform_propagation(
        fixed_image,
        moving_image,
        combined_transform,
        default_value=default_value,
        interp=final_interp,
    )
    registered_image = sitk.Cast(registered_image, moving_image_type)

    return registered_image, combined_transform


def transform_propagation(
    fixed_image,
    moving_image,
    transform,
    structure=False,
    default_value=-1024,
    interp=sitk.sitkNearestNeighbor,
    debug=False,
):
    """
    Transform propagation using ITK

    Args
        fixed_image (sitk.Image)     : the fixed image
        moving_image (sitk.Image)    : the moving image, to be propagated
        transform (sitk.transform)  : the transformation; e.g. VersorRigid3DTransform,
                                      AffineTransform
        structure (bool)            : True if the image is a structure image
        interp (int)                : the interpolation
                                        sitk.sitkNearestNeighbor
                                        sitk.sitkLinear
                                        sitk.sitkBSpline

    Returns
        registered_image (sitk.Image)        : the rigidly registered moving image

    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(interp)
    if structure:
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetDefaultPixelValue(default_value)

    output_image = resampler.Execute(moving_image)

    if structure and interp > 1:
        if debug:
            print(
                "Note: Higher order interpolation on binary mask - using 32-bit floating point output"
            )
        output_image = sitk.Cast(output_image, sitk.sitkFloat32)

        # Safe way to remove dodgy values that can cause issues later
        output_image = sitk.Threshold(output_image, lower=1e-5, upper=100.0)
    else:
        output_image = sitk.Cast(output_image, moving_image.GetPixelID())

    return output_image


def smooth_and_resample(
    image,
    shrink_factor,
    smoothing_sigma,
    isotropic_resample=False,
    resampler=sitk.sitkLinear,
):
    """
    Args:
        image: The image we want to resample.
        shrink_factor: A number greater than one, such that the new image's size is
                       original_size/shrink_factor.
                       If isotropic_resample is True, this will instead define the voxel size (mm)
        smoothing_sigma: Sigma for Gaussian smoothing, this is in physical (image spacing) units,
                         not pixels.
        isotropic_resample: A flag that changes the behaviour to resample the image to isotropic voxels of size (shrink_factor)
    Return:
        Image which is a result of smoothing the input and then resampling it using the given sigma
        and shrink factor.
    """
    if smoothing_sigma > 0:
        # smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigma)
        maximumKernelWidth = int(
            max([8 * smoothing_sigma * i for i in image.GetSpacing()])
        )
        smoothed_image = sitk.DiscreteGaussian(
            image, smoothing_sigma ** 2, maximumKernelWidth
        )
    else:
        smoothed_image = image

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    if isotropic_resample:
        scale_factor = (
            shrink_factor * np.ones_like(image.GetSize()) / np.array(image.GetSpacing())
        )
        new_size = [
            int(sz / float(sf) + 0.5) for sz, sf in zip(original_size, scale_factor)
        ]

    if not isotropic_resample:
        if type(shrink_factor) == list:
            new_size = [
                int(sz / float(sf) + 0.5)
                for sz, sf in zip(original_size, shrink_factor)
            ]
        else:
            new_size = [int(sz / float(shrink_factor) + 0.5) for sz in original_size]

    new_spacing = [
        ((original_sz - 1) * original_spc) / (new_sz - 1)
        for original_sz, original_spc, new_sz in zip(
            original_size, original_spacing, new_size
        )
    ]

    return sitk.Resample(
        smoothed_image,
        new_size,
        sitk.Transform(),
        resampler,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        0.0,
        image.GetPixelID(),
    )


def multiscale_demons(
    registration_algorithm,
    fixed_image,
    moving_image,
    initial_transform=None,
    initial_displacement_field=None,
    shrink_factors=None,
    smoothing_sigmas=None,
    iteration_staging=None,
    isotropic_resample=False,
    return_field=False,
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

    for shrink_factor, smoothing_sigma in reversed(
        list(zip(shrink_factors, smoothing_sigmas))
    ):
        fixed_images.append(
            smooth_and_resample(
                fixed_image,
                shrink_factor,
                smoothing_sigma,
                isotropic_resample=isotropic_resample,
            )
        )
        moving_images.append(
            smooth_and_resample(
                moving_image,
                shrink_factor,
                smoothing_sigma,
                isotropic_resample=isotropic_resample,
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
        initial_displacement_field = sitk.Resample(
            initial_displacement_field, fixed_images[-1]
        )

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

    if return_field:
        return (
            sitk.DisplacementFieldTransform(initial_displacement_field),
            output_displacement_field,
        )
    else:
        return sitk.DisplacementFieldTransform(initial_displacement_field)


def fast_symmetric_forces_demons_registration(
    fixed_image,
    moving_image,
    resolution_staging=[8, 4, 1],
    iteration_staging=[10, 10, 10],
    isotropic_resample=False,
    initial_displacement_field=None,
    smoothing_sigma_factor=1,
    smoothing_sigmas=False,
    default_value=-1024,
    ncores=1,
    structure=False,
    interp_order=2,
    trace=False,
    return_field=False,
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
                                           case resolution_staging is used to define voxel size (mm) per level
        initial_displacement_field (sitk.Image) : Initial displacement field to use
        ncores (int)                    : number of processing cores to use
        structure (bool)                : True if the image is a structure image
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
    if trace:
        registration_method.AddCommand(
            sitk.sitkIterationEvent,
            lambda: deformable_registration_command_iteration(registration_method),
        )

    if not smoothing_sigmas:
        smoothing_sigmas = [i * smoothing_sigma_factor for i in resolution_staging]

    output = multiscale_demons(
        registration_algorithm=registration_method,
        fixed_image=fixed_image,
        moving_image=moving_image,
        shrink_factors=resolution_staging,
        smoothing_sigmas=smoothing_sigmas,
        iteration_staging=iteration_staging,
        isotropic_resample=isotropic_resample,
        initial_displacement_field=initial_displacement_field,
        return_field=return_field,
    )

    if return_field:
        output_transform, deformation_field = output
    else:
        output_transform = output

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(interp_order)

    if structure:
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetDefaultPixelValue(default_value)

    resampler.SetTransform(output_transform)
    registered_image = resampler.Execute(moving_image)

    if structure:
        registered_image = sitk.Cast(registered_image, sitk.sitkFloat32)
        registered_image = sitk.BinaryThreshold(
            registered_image, lowerThreshold=1e-5, upperThreshold=100
        )

    registered_image.CopyInformation(fixed_image)
    registered_image = sitk.Cast(registered_image, moving_image_type)

    if return_field:
        resampled_field = sitk.Resample(deformation_field, fixed_image)
        return registered_image, output_transform, resampled_field
    else:
        return registered_image, output_transform


def apply_field(
    input_image,
    transform,
    structure=False,
    default_value=-1024,
    interp=sitk.sitkNearestNeighbor,
):
    """
    Transform a volume of structure with the given deformation field.

    Args
        input_image (sitk.Image)        : the image to transform
        transform (sitk.Transform)      : the transform to apply to the structure or mask
        structure (bool)  : if true, the input will be treated as a struture, as a volume otherwise
        interp (int)   : the type of interpolation to use, eg. sitk.sitkNearestNeighbor

    Returns
        resampled_image (sitk.Image)    : the transformed image
    """
    input_image_type = input_image.GetPixelIDValue()
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(input_image)

    if structure:
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetDefaultPixelValue(default_value)

    resampler.SetTransform(transform)
    resampler.SetInterpolator(interp)

    resampled_image = resampler.Execute(sitk.Cast(input_image, sitk.sitkFloat32))

    return sitk.Cast(resampled_image, input_image_type)


def bspline_registration(
    fixed_image,
    moving_image,
    moving_structure=False,
    fixed_structure=False,
    options={
        "resolution_staging": [8, 4, 2],
        "smooth_sigmas": [4, 2, 1],
        "sampling_rate": 0.1,
        "optimiser": "LBFGS",
        "metric": "correlation",
        "initial_grid_spacing": 64,
        "grid_scale_factors": [1, 2, 4],
        "interp_order": 3,
        "default_value": -1024,
        "number_of_iterations": 20,
    },
    isotropic_resample=False,
    initial_isotropic_size=1,
    initial_isotropic_smooth_scale=0,
    trace=False,
    ncores=8,
    debug=False,
):
    """
    B-Spline image registration using ITK

    IMPORTANT - THIS IS UNDER ACTIVE DEVELOPMENT

    Args
        fixed_image (sitk.Image) : the fixed image
        moving_image (sitk.Image): the moving image, transformed to match fixed_image
        options (dict)          : registration options
        structure (bool)        : True if the image is a structure image

    Returns
        registered_image (sitk.Image): the rigidly registered moving image
        transform (transform        : the transform, can be used directly with
                                      sitk.ResampleImageFilter

    Notes:
     - smooth_sigmas are relative to resolution staging
        e.g. for image spacing of 1x1x1 mm^3, with smooth sigma=2 and resolution_staging=4, the scale of the Gaussian filter would be 2x4 = 8mm (i.e. 8x8x8 mm^3)

    """

    # Get the settings
    resolution_staging = options["resolution_staging"]
    smooth_sigmas = options["smooth_sigmas"]
    sampling_rate = options["sampling_rate"]
    optimiser = options["optimiser"]
    metric = options["metric"]
    initial_grid_spacing = options["initial_grid_spacing"]
    grid_scale_factors = options["grid_scale_factors"]
    number_of_iterations = options["number_of_iterations"]
    interp_order = options["interp_order"]
    default_value = options["default_value"]

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
            initial_isotropic_size,
            initial_isotropic_smooth_scale,
            isotropic_resample=True,
        )
        moving_image = smooth_and_resample(
            moving_image,
            initial_isotropic_size,
            initial_isotropic_smooth_scale,
            isotropic_resample=True,
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
    if optimiser == "LBFGSB":
        registration.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=number_of_iterations,
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=1024,
            costFunctionConvergenceFactor=1e7,
            trace=trace,
        )
    elif optimiser == "LBFGS":
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
    elif optimiser == "CGLS":
        registration.SetOptimizerAsConjugateGradientLineSearch(
            learningRate=0.05, numberOfIterations=number_of_iterations
        )
        registration.SetOptimizerScalesFromPhysicalShift()
    elif optimiser == "GradientDescent":
        registration.SetOptimizerAsGradientDescent(
            learningRate=5.0,
            numberOfIterations=number_of_iterations,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
        registration.SetOptimizerScalesFromPhysicalShift()
    elif optimiser == "GradientDescentLineSearch":
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
        try:
            number_of_histogram_bins = options["number_of_histogram_bins"]
        except:
            number_of_histogram_bins = 30
        registration.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=number_of_histogram_bins
        )

    registration.SetInterpolator(sitk.sitkLinear)

    # Set sampling
    if type(sampling_rate) == float:
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

    if debug:
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
    if trace:
        registration.AddCommand(
            sitk.sitkIterationEvent,
            lambda: initial_registration_command_iteration(registration),
        )
        registration.AddCommand(
            sitk.sitkMultiResolutionIterationEvent,
            lambda: stage_iteration(registration),
        )

    # Run the registration
    output_transform = registration.Execute(fixed=fixed_image, moving=moving_image)

    # Resample moving image
    registered_image = transform_propagation(
        fixed_image_original,
        moving_image,
        output_transform,
        default_value=default_value,
        interp=interp_order,
    )
    registered_image = sitk.Cast(registered_image, moving_image_type)

    # Return outputs
    return registered_image, output_transform