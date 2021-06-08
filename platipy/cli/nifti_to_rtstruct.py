# Copyright 2020 CSIRO

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

from platipy.dicom.io.nifti_to_rtstruct import convert_nifti


@click.command()
@click.option(
    "--dcm_file",
    "-d",
    required=True,
    help="Reference DICOM file from which header tags will be copied",
)
@click.option("--mask", "-m", multiple=True, required=True, help="Mask pairs with name,filename")
@click.option("--out_rt_filename", "-o", required=True, help="Name of RT struct output")
def click_command(dcm_file, mask, out_rt_filename):
    """
    Convert Nifti masks to Dicom RTStruct
    """

    convert_nifti(dcm_file, mask, out_rt_filename)


if __name__ == "__main__":
    click_command()  # pylint: disable=no-value-for-parameter
