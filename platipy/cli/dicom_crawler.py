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

import sys
import click

from loguru import logger

from platipy.dicom.io.crawl import (
    process_dicom_directory,
)

logger.remove()
logger.add(sys.stderr, level="DEBUG")


@click.command()
@click.option(
    "--input_dir",
    "-i",
    required=True,
    type=click.Path(),
    help="Input DICOM directory. This should be at the same level as the parent field "
    "(default=PatientName).",
)
@click.option(
    "--output_dir",
    "-o",
    default="./",
    show_default=True,
    required=False,
    type=click.Path(),
    help="Output directory. A folder structure will be created at this location.",
)
@click.option(
    "--sort_by",
    "-b",
    default="PatientName",
    help="DICOM tag to sort at the highest level.",
    show_default=True,
)
@click.option(
    "--image_format",
    default="{parent_sorting_data}_{study_uid_index}_{Modality}_{image_desc}_{SeriesNumber}",
    help="Format for output images. There are three special options that can be used: "
    "parent_sorting_data (same as sort_by option), study_uid_index (a counter for distinct DICOM "
    "studies), image_desc (info from DICOM header, more nicely formatted). Additionally, any "
    "DICOM header tag can be used (e.g. Modality, SeriesNumber, AcquisitionData). Any DICOM "
    "header tag that doesn't exist will return a 0.",
    show_default=True,
)
@click.option(
    "--structure_format",
    default="{parent_sorting_data}_{study_uid_index}_{Modality}_{structure_name}",
    help="Format for output structures. Any of the options for images can be used, as "
    "well as: structure_name",
    show_default=True,
)
@click.option(
    "--dose_format",
    default="{parent_sorting_data}_{study_uid_index}_{DoseSummationType}",
    show_default=True,
    help="Format for output radiotherapy dose distributions.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite files if they exist.",
    show_default=True,
)
@click.option(
    "--file_suffix",
    default=".nii.gz",
    help="Output file suffix. Defines the file type.",
    show_default=True,
)
@click.option(
    "--short_description",
    "-s",
    is_flag=True,
    default=False,
    show_default=True,
    help="Use less verbose descriptions for DICOM images.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    show_default=True,
    help="Print more information while running.",
)
def click_command(
    input_dir,
    output_dir,
    sort_by,
    image_format,
    structure_format,
    dose_format,
    overwrite,
    file_suffix,
    short_description,
    verbose,
):
    """
    DICOM DIRECTORY CRAWLER

    This tool makes it easier to bulk-convert DICOM files into other formats (default NifTI).
    There are quite a lot of options, but most do not need to be changed.

    You need to provide the input directory (-i), from which the crawler will recursively search
    through.

    You might also like to change the naming format (using --image_format and --structure_format).
    The default is quite long, better suited for datasets with lots of imaging for a single
    patient.

    Some examples:

      [simple] --image_format {parent_sorting_data}

      [compact]  --image_format {parent_sorting_data}_{study_uid_index}

    You can separate series using different values (--sort_by ).
    This would typically be PatientName, or PatientID, although any DICOM header tag is allowed.

    I hope you find this tool useful!

    If you have any feedback, let us know on github.com/pyplati/platipy

    """
    logger.info("########################")
    logger.info(" Running DICOM crawler ")
    logger.info("########################")

    process_dicom_directory(
        input_dir,
        parent_sorting_field=sort_by,
        output_image_name_format=image_format,
        output_structure_name_format=structure_format,
        output_dose_name_format=dose_format,
        return_extra=(not short_description),
        output_directory=output_dir,
        output_file_suffix=file_suffix,
        overwrite_existing_files=overwrite,
        write_to_disk=True,
        verbose=verbose,
    )

    logger.info("########################")
    logger.info(" DICOM crawler complete")
    logger.info("########################")


if __name__ == "__main__":
    click_command()  # pylint: disable=no-value-for-parameter
