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

from platipy.dicom.dicom_directory_crawler.conversion_utils import (
    process_dicom_directory,
    write_output_data_to_disk
)

logger.remove()
logger.add(sys.stderr, level="DEBUG")

@click.command()
@click.option(
    "--input_dir",
    '-i',
    required=True,
    type=click.Path(),
    help="Input DICOM directory. This should be at the same level as the parent field (default=PatientName)."
    )
@click.option(
    "--output_dir",
    '-o',
    default="./",
    required=False,
    type=click.Path(),
    help="Output directory. A folder structure will be created at this location."
    )
@click.option(
    "--sort_by",
    "-b",
    default="PatientName",
    help="DICOM tag to sort at the highest level."
    )
@click.option(
    "--image_format",
    default="{parent_sorting_data}_{study_uid_index}_{image_modality}_{image_desc}_{series_num}",
    help="Format for output images. Any of the following options can be used: parent_sorting_data, study_uid_index, image_modality, image_desc, series_num, acq_number, acq_date"
    )
@click.option(
    "--structure_format",
    default="{parent_sorting_data}_{study_uid_index}_{image_modality}_{structure_name}",
    help="Format for output structures. Any of the options for images can be used, as well as: structure_name"
    )
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite files if they exist."
    )
@click.option(
    "--file_suffix",
    default=".nii.gz",
    help="Output file suffix. Defines the file type."
    )
@click.option(
    "--short_description",
    "-s",
    is_flag=True,
    default=False,
    help="Use less verbose descriptions for DICOM images."
    )
def click_command(input_dir, output_dir, sort_by, image_format, structure_format, overwrite, file_suffix, short_description):

    logger.info("########################")
    logger.info(" Running DICOM crawler ")
    logger.info("########################")

    output_data_dict = process_dicom_directory( input_dir,
                                                parent_sorting_field=sort_by,
                                                output_image_name_format = image_format,
                                                output_structure_name_format = structure_format,
                                                return_extra=(not short_description))

    write_output_data_to_disk(  output_data_dict,
                                output_directory = output_dir,
                                output_file_suffix = file_suffix,
                                overwrite_existing_files = overwrite)

    logger.info("########################")
    logger.info(" DICOM crawler complete")
    logger.info("########################")

if __name__ == "__main__":
    click_command()  # pylint: disable=no-value-for-parameter