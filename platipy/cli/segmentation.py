#!/usr/bin/env python

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

import os
import sys
import logging
import json

import click

import SimpleITK as sitk

from platipy.imaging.projects.bronchus.run import (
    run_bronchus_segmentation,
    BRONCHUS_SETTINGS_DEFAULTS,
)

logger = logging.getLogger(__name__)

segmentation_algorithms = {}

try:
    from platipy.imaging.projects.cardiac.run import (
        run_hybrid_segmentation,
        HYBRID_SETTINGS_DEFAULTS,
    )

    segmentation_algorithms["cardiac"] = {
        "algorithm": run_hybrid_segmentation,
        "default_settings": HYBRID_SETTINGS_DEFAULTS,
    }
except ImportError:
    logger.warning(
        "Unable to provide cardiac segmentation. Be sure to install platipy with "
        "cardiac extras: pip install 'platipy[cardiac]'"
    )

segmentation_algorithms["bronchus"] = {
    "algorithm": run_bronchus_segmentation,
    "default_settings": BRONCHUS_SETTINGS_DEFAULTS,
}


@click.command()
@click.argument("algorithm", nargs=1, type=click.Choice(segmentation_algorithms.keys()))
@click.argument("input_path", nargs=1, type=click.Path(), required=False)
@click.option(
    "--config",
    "-c",
    required=False,
    type=click.Path(),
    help="Path to JSON file containing algorithm settings",
)
@click.option(
    "--default",
    "-d",
    is_flag=True,
    help="Print the default configuration for the selected algorithm",
)
@click.option(
    "--output",
    "-o",
    required=False,
    type=click.Path(),
    help="Path to directory for output",
)
def click_command(algorithm, input_path, config, default, output):
    """
    Run an auto-segmentation on an input image. Choose which algorithm to run and pass the path to
    the Nitfti input image OR a directory containing a DICOM series.

    Optionally, pass in a configuration file for the segmentation algorithm. Output the default
    configuration for an algorithm using the --default flag.
    """

    algorithm_config = segmentation_algorithms[algorithm]["default_settings"]

    if default:
        print(json.dumps(algorithm_config, indent=4))
        return

    # If we get to here but no input_path was set, we need to inform the user
    if not input_path:
        print("Supply the path the the Nifti Image OR DICOM series to process")
        return

    print(f"Running {algorithm} segmentation")

    if config:
        with open(config, "r") as file_obj:
            algorithm_config = json.load(file_obj)

    read_path = input_path
    if os.path.isdir(input_path):
        # If it's a directory then read as a DICOM series
        read_path = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(input_path)

    image = sitk.ReadImage(read_path)

    # Run the algorithm
    results = segmentation_algorithms[algorithm]["algorithm"](image, algorithm_config)

    # If results is a tuple, the algorithm is returning other items as well as the semgmentation.
    # We are only interested in the segmentation here (the first element of the tuple)
    if isinstance(results, tuple):
        results = results[0]

    # Save the output to the output directory
    if not output:
        output = "."
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    for result in results:
        sitk.WriteImage(results[result], os.path.join(output, f"{result}.nii.gz"))


if __name__ == "__main__":

    click_command()  # pylint: disable=no-value-for-parameter
