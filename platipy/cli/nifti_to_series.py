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

import click

import SimpleITK as sitk

from platipy.dicom.io.nifti_to_series import convert_nifti_to_dicom_series


@click.command()
@click.option(
    "--dcm",
    "-d",
    required=True,
    help="Reference DICOM folder containing series from which header tags will be copied",
)
@click.option("--image", "-i", required=True, help="Nifti Image file to convert to Dicom series")
@click.option(
    "--tag", "-t", multiple=True, help="Override Dicom Tags by providing key:value pairs"
)
@click.option(
    "--output_directory",
    "-o",
    help="Directory in which Dicom series files will be generated",
    default=".",
    show_default=True,
)
def click_command(dcm, image, tag, output_directory):
    """
    Convert a Nifti image to a Dicom image series
    """

    tags = [t.split(":") for t in tag]
    convert_nifti_to_dicom_series(
        sitk.ReadImage(image), dcm, tag_overrides=tags, output_directory=output_directory
    )


if __name__ == "__main__":
    click_command()  # pylint: disable=no-value-for-parameter
