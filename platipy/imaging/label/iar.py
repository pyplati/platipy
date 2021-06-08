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

from loguru import logger

import numpy as np
import SimpleITK as sitk

from scipy.optimize import curve_fit
from scipy.ndimage import filters
from scipy.stats import norm as scipy_norm

from platipy.imaging.label.fusion import combine_labels, process_probability_image

from platipy.imaging.label.projection import (
    evaluate_distance_on_surface,
    evaluate_distance_to_reference,
    regrid_spherical_data,
)


def median_absolute_deviation(data, axis=None):
    """Median Absolute Deviation: a "Robust" version of standard deviation.
    Indices variabililty of the sample.
    https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    return np.median(np.abs(data - np.median(data, axis=axis)), axis=axis)


def gaussian_curve(x, a, m, s):
    """Returns a Gaussian (normal) curve

    Args:
        x (np.ndarray): values to sample the normal curve
        a (float): magnitude
        m (float): location (mean)
        s (float): scale (standard deviation)

    Returns:
        np.ndarray: sampled values along the normal curve
    """
    return a * scipy_norm.pdf(x, loc=m, scale=s)


def run_iar(
    atlas_set,
    reference_structure,
    smooth_distance_maps=False,
    smooth_sigma=1,
    z_score_statistic="MAD",
    outlier_method="IQR",
    min_best_atlases=10,
    outlier_factor=1.5,
    iteration=0,
    single_step=False,
    project_on_sphere=False,
    label="DIR",
):
    """
    Perform iterative atlas removal on the atlas_set
    """

    if iteration == 0:
        # Run some checks in the data
        # If there is a MAJOR error we need to check

        # Begin the process
        logger.info("Iterative atlas removal: ")
        logger.info("  Beginning process")

    # Get remaining case identifiers to loop through
    remaining_id_list = list(atlas_set.keys())

    # Generate the surface projections
    #   1. Set the consensus surface using the reference volume
    probability_label = combine_labels(atlas_set, reference_structure, label=label)[
        reference_structure
    ]

    # Modify resolution for better statistics
    if project_on_sphere:
        if len(remaining_id_list) < 12:
            logger.info("  Less than 12 atlases, resolution set: 3x3 sqr deg")
            resolution = 3
        elif len(remaining_id_list) < 7:
            logger.info("  Less than 7 atlases, resolution set: 6x6 sqr deg")
            resolution = 6
        else:
            resolution = 1
    else:
        if len(remaining_id_list) < 12:
            logger.info("  Less than 12 atlases, resample factor set: 5")
            resample_factor = 5
        elif len(remaining_id_list) < 7:
            logger.info("  Less than 7 atlases, resolution set: 6x6 sqr deg")
            resample_factor = 10
        else:
            resample_factor = 1

    g_val_list = []
    logger.info("  Calculating surface distance maps: ")
    for test_id in remaining_id_list:
        logger.info("    {0}".format(test_id))
        #   2. Calculate the distance from the surface to the consensus surface

        test_volume = atlas_set[test_id][label][reference_structure]

        # This next step ensures non-binary labels are treated properly
        # We use 0.1 to capture the outer edge of the test delineation, if it is probabilistic
        test_volume = process_probability_image(test_volume, 0.1)

        if project_on_sphere:
            reference_volume = process_probability_image(probability_label, threshold=0.999)
            # note: we use a threshold slightly below 1 to ensure the consensus (reference) volume
            # is a suitable binary volume

            # Compute the reference distance map
            reference_distance_map = sitk.Abs(
                sitk.SignedMaurerDistanceMap(
                    reference_volume, squaredDistance=False, useImageSpacing=True
                )
            )

            # Compute the distance to test surfaces, across the surface of the reference
            theta, phi, values = evaluate_distance_on_surface(
                reference_distance_map, test_volume, reference_as_distance_map=True
            )

            _, _, g_vals = regrid_spherical_data(theta, phi, values, resolution=resolution)

            g_val_list.append(g_vals)
        else:
            reference_volume = process_probability_image(probability_label, threshold=0.95)
            # note: we use a threshold slightly below 1 to ensure the consensus (reference) volume
            # is a suitable binary volume we have the flexibility to modify the reference volume
            # when we do not use spherical projection a larger surface means more evaluations and
            # better statistics, so we prefer a lower threshold but not too low,
            # or it may include some errors

            # Compute distance to reference, from the test volume
            values = evaluate_distance_to_reference(
                reference_volume, test_volume, resample_factor=resample_factor
            )

            g_val_list.append(values)

    q_results = {}

    for i, (test_id, g_vals) in enumerate(zip(remaining_id_list, g_val_list)):

        g_val_list_test = g_val_list[:]
        g_val_list_test.pop(i)

        if project_on_sphere and smooth_distance_maps:
            g_vals = filters.gaussian_filter(g_vals, sigma=smooth_sigma, mode="wrap")

        #       b) i] Compute the Z-scores over the projected surface
        if z_score_statistic.lower() == "std":
            g_val_mean = np.mean(g_val_list_test, axis=0)
            g_val_std = np.std(g_val_list_test, axis=0)

            if np.any(g_val_std == 0):
                logger.info("    Std Dev zero count: {0}".format(np.sum(g_val_std == 0)))
                g_val_std[g_val_std == 0] = g_val_std.mean()

            z_score_vals_array = (g_vals - g_val_mean) / g_val_std

        elif z_score_statistic.lower() == "mad":
            g_val_median = np.median(g_val_list_test, axis=0)
            g_val_mad = 1.4826 * median_absolute_deviation(g_val_list_test, axis=0)

            if np.any(~np.isfinite(g_val_mad)):
                logger.info("Error in MAD")
                logger.info(g_val_mad)

            if np.any(g_val_mad == 0):
                logger.info("    MAD zero count: {0}".format(np.sum(g_val_mad == 0)))
                g_val_mad[g_val_mad == 0] = np.median(g_val_mad)

            z_score_vals_array = (g_vals - g_val_median) / g_val_mad

        else:
            logger.error(" Error!")
            logger.error(" z_score must be one of: MAD, STD")
            sys.exit()

        z_score_vals = np.ravel(z_score_vals_array)

        logger.debug("      [{0}] Statistics of mZ-scores".format(test_id))
        logger.debug("        Min(Z)    = {0:.2f}".format(z_score_vals.min()))
        logger.debug("        Q1(Z)     = {0:.2f}".format(np.percentile(z_score_vals, 25)))
        logger.debug("        Mean(Z)   = {0:.2f}".format(z_score_vals.mean()))
        logger.debug("        Median(Z) = {0:.2f}".format(np.percentile(z_score_vals, 50)))
        logger.debug("        Q3(Z)     = {0:.2f}".format(np.percentile(z_score_vals, 75)))
        logger.debug("        Max(Z)    = {0:.2f}\n".format(z_score_vals.max()))

        # Calculate excess area from Gaussian: the Q-metric
        bins = np.linspace(-15, 15, 501)
        z_density, bin_edges = np.histogram(z_score_vals, bins=bins, density=True)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

        try:
            popt, _ = curve_fit(  # pylint: disable=unbalanced-tuple-unpacking
                f=gaussian_curve, xdata=bin_centers, ydata=z_density
            )

            z_ideal = gaussian_curve(bin_centers, *popt)
            z_diff = np.abs(z_density - z_ideal)
        except (RuntimeError, ValueError):
            logger.debug("IAR couldnt fit curve, estimating with sampled statistics.")
            z_ideal = gaussian_curve(bin_centers, a=1, m=z_density.mean(), s=z_density.std())
            z_diff = np.abs(z_density - z_ideal)

        # Integrate to get the q_value
        q_value = np.trapz(z_diff * np.abs(bin_centers) ** 2, bin_centers)
        q_results[test_id] = np.float64(q_value)

    # Exclude (at most) the worst 3 atlases for outlier detection
    # With a minimum number, this helps provide more robust estimates at low numbers
    result_list = list(q_results.values())
    result_list = [r for r in result_list if ~np.isnan(r) and np.isfinite(r)]
    best_results = np.sort(result_list)[: max([min_best_atlases, len(result_list) - 3])]

    if outlier_method.lower() == "iqr":
        outlier_limit = np.percentile(best_results, 75, axis=0) + outlier_factor * np.subtract(
            *np.percentile(best_results, [75, 25], axis=0)
        )
    elif outlier_method.lower() == "std":
        outlier_limit = np.mean(best_results, axis=0) + outlier_factor * np.std(
            best_results, axis=0
        )
    else:
        logger.error(" Error!")
        logger.error(" outlier_method must be one of: IQR, STD")
        sys.exit()

    logger.info("  Analysing results")
    logger.info("   Outlier limit: {0:06.3f}".format(outlier_limit))
    keep_id_list = []

    logger.info(
        "{0},{1},{2},{3:.4g}\n".format(
            iteration,
            " ".join(remaining_id_list),
            " ".join(["{0:.4g}".format(i) for i in list(q_results.values())]),
            outlier_limit,
        )
    )

    for idx, result in q_results.items():

        accept = result <= outlier_limit

        logger.info(
            "      {0}: Q = {1:06.3f} [{2}]".format(
                idx, result, {True: "KEEP", False: "REMOVE"}[accept]
            )
        )

        if accept:
            keep_id_list.append(idx)

    if len(keep_id_list) < len(remaining_id_list):
        logger.info("\n  Step {0} Complete".format(iteration))
        logger.info("  Num. Removed = {0} --\n".format(len(remaining_id_list) - len(keep_id_list)))

        iteration += 1
        atlas_set_new = {i: atlas_set[i] for i in keep_id_list}

        if single_step:
            return atlas_set_new

        return run_iar(
            atlas_set=atlas_set_new,
            reference_structure=reference_structure,
            smooth_distance_maps=smooth_distance_maps,
            smooth_sigma=smooth_sigma,
            z_score_statistic=z_score_statistic,
            outlier_method=outlier_method,
            min_best_atlases=min_best_atlases,
            outlier_factor=outlier_factor,
            iteration=iteration,
            project_on_sphere=project_on_sphere,
            label=label,
        )

    logger.info("  End point reached. Keeping:\n   {0}".format(keep_id_list))

    return atlas_set
