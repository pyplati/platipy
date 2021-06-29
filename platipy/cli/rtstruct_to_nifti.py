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

from platipy.dicom.io.rtstruct_to_nifti import convert_rtstruct


@click.command()
@click.option(
    "--dcm_img",
    "-i",
    required=True,
    help="Directory containing the image series linked to the contour to convert",
)
@click.option(
    "--dcm_rt_file",
    "-r",
    required=True,
    help="Dicom RTStruct file containing the contours to convert",
)
@click.option("--prefix", "-p", default="Struct_", help="Prefix for output files (e.g. Case_01_")
@click.option(
    "--output_dir",
    "-od",
    default=".",
    help="Directory in which to place converted files",
    show_default=True,
)
@click.option("--output_img", "-oi", help="Output name of converted image series")
@click.option(
    "--spacing",
    help="DICOM image spacing override with format x,y,z (0 indicates to leave as is, e.g. 0,0,3)",
)
def click_command(dcm_img, dcm_rt_file, prefix, output_dir, output_img, spacing):
    """
    Convert a DICOM RTSTRUCT file to Nifti format.
    """

    convert_rtstruct(
        dcm_img,
        dcm_rt_file,
        prefix=prefix,
        output_dir=output_dir,
        output_img=output_img,
        spacing=spacing,
    )


if __name__ == "__main__":

    click_command()  # pylint: disable=no-value-for-parameter
