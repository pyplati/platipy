"""
Provides tools to perform registration for use within atlas based segmentation algorithms.
"""

import sys

import SimpleITK as sitk


def initial_registration(
    fixed_image,
    moving_image,
    moving_structure=False,
    fixed_structure=False,
    options=None,
    trace=False,
    reg_method="Rigid",
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
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    if not options:
        options = {"shrinkFactors": [8, 2, 1], "smoothSigmas": [4, 2, 0], "samplingRate": 0.1}

    # Get the options
    shrink_factors = options["shrinkFactors"]
    smooth_sigmas = options["smoothSigmas"]
    sampling_rate = options["samplingRate"]

    if reg_method == "Rigid":
        # Select the rigid transform
        transform = sitk.VersorRigid3DTransform()
    elif reg_method == "Affine":
        # Select the affine transform
        transform = sitk.AffineTransform(3)
    elif reg_method == "Translation":
        # Select the translation transform
        transform = sitk.TranslationTransform(3)
    else:
        print("[ERROR] Registration method must be Rigid, Affine or Translation.")
        sys.exit()

    # Set up image registration method
    registration = sitk.ImageRegistrationMethod()

    registration.SetShrinkFactorsPerLevel(shrink_factors)
    registration.SetSmoothingSigmasPerLevel(smooth_sigmas)
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=512,
        maximumNumberOfCorrections=50,
        maximumNumberOfFunctionEvaluations=1024,
        costFunctionConvergenceFactor=1e7,
        trace=trace,
    )

    registration.SetMetricAsMeanSquares()
    registration.SetInterpolator(sitk.sitkLinear)  # Perhaps a small gain in improvement
    registration.SetMetricSamplingPercentage(sampling_rate)
    registration.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.REGULAR)

    if moving_structure:
        registration.SetMetricMovingMask(moving_structure)

    if fixed_structure:
        registration.SetMetricFixedMask(fixed_structure)

    initializer = sitk.CenteredTransformInitializerFilter()
    initializer.GeometryOn()
    initial_transform = initializer.Execute(fixed_image, moving_image, transform)

    registration.SetInitialTransform(initial_transform)
    output_transform = registration.Execute(fixed=fixed_image, moving=moving_image)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(output_transform)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(-1024)

    registered_image = resampler.Execute(moving_image)

    return registered_image, output_transform


def transform_propagation(
    fixed_image, moving_image, transform, structure=False, interp=sitk.sitkNearestNeighbor
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
        resampler.SetDefaultPixelValue(-1024)

    output_image = resampler.Execute(moving_image)

    if structure and interp > 1:
        print(
            "Note: Higher order interpolation on binary mask - using 32-bit floating point output."
        )
        output_image = sitk.Cast(output_image, sitk.sitkFloat32)

        # Safe way to remove dodgy values that can cause issues later
        output_image = sitk.Threshold(output_image, lower=1e-5, upper=100.0)
    else:
        output_image = sitk.Cast(output_image, moving_image.GetPixelID())

    return output_image


def smooth_and_resample(image, shrink_factor, smoothing_sigma):
    """
    Args:
        image: The image we want to resample.
        shrink_factor: A number greater than one, such that the new image's size is
                       original_size/shrink_factor.
        smoothing_sigma: Sigma for Gaussian smoothing, this is in physical (image spacing) units,
                         not pixels.
    Return:
        Image which is a result of smoothing the input and then resampling it using the given sigma
        and shrink factor.
    """
    if smoothing_sigma > 0:
        smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigma)
    else:
        smoothed_image = image

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(sz / float(shrink_factor) + 0.5) for sz in original_size]
    new_spacing = [
        ((original_sz - 1) * original_spc) / (new_sz - 1)
        for original_sz, original_spc, new_sz in zip(original_size, original_spacing, new_size)
    ]
    return sitk.Resample(
        smoothed_image,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
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
    shrink_factors=None,
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
        shrink_factors: Shrink factors relative to the original image's size.
        smoothing_sigmas: Amount of smoothing which is done prior to resmapling the image using the
                          given shrink factor. These are in physical (image spacing) units.
    Returns:
        SimpleITK.DisplacementFieldTransform
    """
    # Create image pyramid.
    fixed_images = []
    moving_images = []
    if shrink_factors:
        for shrink_factor, smoothing_sigma in reversed(
            list(zip(shrink_factors, smoothing_sigmas))
        ):
            fixed_images.append(smooth_and_resample(fixed_image, shrink_factor, smoothing_sigma))
            moving_images.append(smooth_and_resample(moving_image, shrink_factor, smoothing_sigma))

    # Create initial displacement field at lowest resolution.
    # Currently, the pixel type is required to be sitkVectorFloat64 because of a constraint imposed
    # by the Demons filters.
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
        initial_displacement_field = sitk.Image(
            fixed_images[-1].GetWidth(),
            fixed_images[-1].GetHeight(),
            fixed_images[-1].GetDepth(),
            sitk.sitkVectorFloat64,
        )
        initial_displacement_field.CopyInformation(fixed_images[-1])

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
    return sitk.DisplacementFieldTransform(initial_displacement_field)


def command_iteration(method):
    """
    Utility function to print information during demons registration
    """
    print("{0:3} = {1:10.5f}".format(method.GetElapsedIterations(), method.GetMetric()))


def fast_symmetric_forces_demons_registration(
    fixed_image,
    moving_image,
    resolution_staging=[8, 4, 1],
    iteration_staging=[10, 10, 10],
    smoothing_sigma_factor=1,
    ncores=1,
    structure=False,
    interp_order=2,
    trace=False,
):
    """
    Deformable image propagation using Fast Symmetric-Forces Demons

    Args
        fixed_image (sitk.Image)        : the fixed image
        moving_image (sitk.Image)       : the moving image, to be deformable registered (must be in
                                          the same image space)
        resolution_staging (list[int])   : down-sampling factor for each resolution level
        iteration_staging (list[int])    : number of iterations for each resolution level
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
    """

    # Cast to floating point representation, if necessary
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
            sitk.sitkIterationEvent, lambda: command_iteration(registration_method)
        )

    output_transform = multiscale_demons(
        registration_algorithm=registration_method,
        fixed_image=fixed_image,
        moving_image=moving_image,
        shrink_factors=resolution_staging,
        smoothing_sigmas=[i * smoothing_sigma_factor for i in resolution_staging],
        iteration_staging=iteration_staging,
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(interp_order)

    if structure:
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetDefaultPixelValue(-1024)

    resampler.SetTransform(output_transform)
    registered_image = resampler.Execute(moving_image)

    if structure:
        registered_image = sitk.Cast(registered_image, sitk.sitkFloat32)
        registered_image = sitk.Threshold(registered_image, lower=1e-5, upper=100)

    registered_image.CopyInformation(fixed_image)

    return registered_image, output_transform


def apply_field(input_image, transform, structure=False, interp=sitk.sitkNearestNeighbor):
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


    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(input_image)

    if structure:
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetDefaultPixelValue(-1024)

    resampler.SetTransform(transform)
    resampler.SetInterpolator(interp)

    resampled_image = resampler.Execute(sitk.Cast(input_image, sitk.sitkFloat32))

    return resampled_image
