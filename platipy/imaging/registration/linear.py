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

import SimpleITK as sitk

from platipy.imaging.registration.utils import (
    apply_transform,
    registration_command_iteration,
)


def alignment_registration(fixed_image, moving_image, moments=True):
    """
    A simple registration procedure that can align images in a single step.
    Uses the image centres-of-mass (and optionally second moments) to
    estimate the shift (and rotation) needed for alignment.

    Args:
        fixed_image ([SimpleITK.Image]): The fixed (target/primary) image.
        moving_image ([SimpleITK.Image]): The moving (secondary) image.
        moments (bool, optional): Option to align images using the second moment. Defaults to True.

    Returns:
        [SimpleITK.Image]: The registered moving (secondary) image.
        [SimleITK.Transform]: The linear transformation.
    """

    moving_image_type = moving_image.GetPixelIDValue()
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image, sitk.VersorRigid3DTransform(), moments
    )
    aligned_image = sitk.Resample(moving_image, fixed_image, initial_transform)
    aligned_image = sitk.Cast(aligned_image, moving_image_type)
    return aligned_image, initial_transform


def linear_registration(
    fixed_image,
    moving_image,
    fixed_structure=None,
    moving_structure=None,
    reg_method="similarity",
    metric="mean_squares",
    optimiser="gradient_descent",
    shrink_factors=[8, 2, 1],
    smooth_sigmas=[4, 2, 0],
    sampling_rate=0.25,
    final_interp=2,
    number_of_iterations=50,
    default_value=None,
    verbose=False,
):
    """
    Initial linear registration between two images.
    The images are not required to be in the same space.
    There are several transforms available, with options for the metric and optimiser to be used.
    Note the default_value, which should be set to match the image modality.

    Args:
        fixed_image ([SimpleITK.Image]): The fixed (target/primary) image.
        moving_image ([SimpleITK.Image]): The moving (secondary) image.
        fixed_structure (bool, optional): If defined, a binary SimpleITK.Image used to mask metric
                                          evaluation for the moving image. Defaults to False.
        moving_structure (bool, optional): If defined, a binary SimpleITK.Image used to mask metric
                                           evaluation for the fixed image. Defaults to False.
        reg_method (str, optional): The linear transformtation model to be used for image
                                    registration.
                                    Available options:
                                     - translation
                                     - rigid
                                     - similarity
                                     - affine
                                     - scale
                                     - scaleversor
                                     - scaleskewversor
                                    Defaults to "Similarity".
        metric (str, optional): The metric to be optimised during image registration.
                                Available options:
                                 - correlation
                                 - mean_squares
                                 - mattes_mi
                                 - joint_hist_mi
                                Defaults to "mean_squares".
        optimiser (str, optional): The optimiser algorithm used for image registration.
                                   Available options:
                                    - lbfgsb
                                      (limited-memory Broyden–Fletcher–Goldfarb–Shanno (bounded).)
                                    - gradient_descent
                                    - gradient_descent_line_search
                                   Defaults to "gradient_descent".
        shrink_factors (list, optional): The multi-resolution downsampling factors.
                                         Defaults to [8, 2, 1].
        smooth_sigmas (list, optional): The multi-resolution smoothing kernel scale (Gaussian).
                                        Defaults to [4, 2, 0].
        sampling_rate (float, optional): The fraction of voxels sampled during each iteration.
                                         Defaults to 0.25.
        ants_radius (int, optional): Used is the metric is set as "ants_radius". Defaults to 3.
        final_interp (int, optional): The final interpolation order. Defaults to 2 (linear).
        number_of_iterations (int, optional): Number of iterations in each multi-resolution step.
                                              Defaults to 50.
        default_value (int, optional): Default voxel value. Defaults to 0 unless image is CT-like.
        verbose (bool, optional): Print image registration process information. Defaults to False.

    Returns:
        [SimpleITK.Image]: The registered moving (secondary) image.
        [SimleITK.Transform]: The linear transformation.
    """

    # Re-cast
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)

    moving_image_type = moving_image.GetPixelIDValue()
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

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

    if metric.lower() == "correlation":
        registration.SetMetricAsCorrelation()
    elif metric.lower() == "mean_squares":
        registration.SetMetricAsMeanSquares()
    elif metric.lower() == "mattes_mi":
        registration.SetMetricAsMattesMutualInformation()
    elif metric.lower() == "joint_hist_mi":
        registration.SetMetricAsJointHistogramMutualInformation()
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

    if isinstance(reg_method, str):
        if reg_method.lower() == "translation":
            registration.SetInitialTransform(sitk.TranslationTransform(3))
        elif reg_method.lower() == "similarity":
            registration.SetInitialTransform(sitk.Similarity3DTransform())
        elif reg_method.lower() == "affine":
            registration.SetInitialTransform(sitk.AffineTransform(3))
        elif reg_method.lower() == "rigid":
            registration.SetInitialTransform(sitk.VersorRigid3DTransform())
        elif reg_method.lower() == "scale":
            registration.SetInitialTransform(sitk.ScaleTransform(3))
        elif reg_method.lower() == "scaleversor":
            registration.SetInitialTransform(sitk.ScaleVersor3DTransform())
        elif reg_method.lower() == "scaleskewversor":
            registration.SetInitialTransform(sitk.ScaleSkewVersor3DTransform())
        else:
            raise ValueError(
                "You have selected a registration method that does not exist.\n Please select from"
                " Translation, Similarity, Affine, Rigid, ScaleVersor, ScaleSkewVersor"
            )
    elif isinstance(
        reg_method,
        (
            sitk.CompositeTransform,
            sitk.Transform,
            sitk.TranslationTransform,
            sitk.Similarity3DTransform,
            sitk.AffineTransform,
            sitk.VersorRigid3DTransform,
            sitk.ScaleVersor3DTransform,
            sitk.ScaleSkewVersor3DTransform,
        ),
    ):
        registration.SetInitialTransform(reg_method)
    else:
        raise ValueError(
            "'reg_method' must be either a string (see docs for acceptable registration names), "
            "or a custom sitk.CompositeTransform."
        )

    if optimiser.lower() == "lbfgsb":
        registration.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=number_of_iterations,
            maximumNumberOfCorrections=50,
            maximumNumberOfFunctionEvaluations=1024,
            costFunctionConvergenceFactor=1e7,
            trace=verbose,
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

    if verbose:
        registration.AddCommand(
            sitk.sitkIterationEvent,
            lambda: registration_command_iteration(registration),
        )

    output_transform = registration.Execute(fixed=fixed_image, moving=moving_image)
    # Combine initial and optimised transform
    combined_transform = sitk.CompositeTransform([initial_transform, output_transform])

    # Try to find default value
    if default_value is None:
        default_value = 0

        # Test if image is CT-like
        if sitk.GetArrayViewFromImage(moving_image).min() <= -1000:
            default_value = -1000

    registered_image = apply_transform(
        input_image=moving_image,
        reference_image=fixed_image,
        transform=combined_transform,
        default_value=default_value,
        interpolator=final_interp,
    )

    registered_image = sitk.Cast(registered_image, moving_image_type)

    return registered_image, combined_transform
