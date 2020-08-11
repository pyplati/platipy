"""Provides a command line interface to use the auto-segmentation tools from the command line
"""

import os
import json

import click

import SimpleITK as sitk

from platipy.segmentation.bronchus.run import run_bronchus_segmentation, BRONCHUS_SETTINGS_DEFAULTS
from platipy.segmentation.cardiac.run import run_cardiac_segmentation, CARDIAC_SETTINGS_DEFAULTS

segmentation_algorithms = {
    "cardiac": {
        "algorithm": run_cardiac_segmentation,
        "default_settings": CARDIAC_SETTINGS_DEFAULTS,
    },
    "bronchus": {
        "algorithm": run_bronchus_segmentation,
        "default_settings": BRONCHUS_SETTINGS_DEFAULTS,
    },
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
def click_command(algorithm, input_path, config, default):
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

    segmentation_algorithms[algorithm]["algorithm"](image, algorithm_config)


if __name__ == "__main__":

    click_command()  # pylint: disable=no-value-for-parameter
