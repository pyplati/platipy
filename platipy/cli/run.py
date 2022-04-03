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

from loguru import logger

from platipy.cli import (
    dicom_crawler,
    segmentation,
    nifti_to_rtstruct,
    rtstruct_to_nifti,
    nifti_to_series,
    tcia_download,
)

tools = {
    "dicom_crawler": dicom_crawler.click_command,
    "segmentation": segmentation.click_command,
    "nifti_to_rtstruct": nifti_to_rtstruct.click_command,
    "rtstruct_to_nifti": rtstruct_to_nifti.click_command,
    "nifti_to_series": nifti_to_series.click_command,
    "tcia-download": tcia_download.click_command,
}

# If backend tools are installed, then provide manage tools
try:
    from platipy.backend.manage import cli

    tools["manage"] = cli
except ImportError:
    logger.info(
        "PlatiPy Backend requirements not installed. Install the requirements listed "
        "at: https://github.com/pyplati/platipy/blob/master/requirements-backend.txt to use "
        "backend service functionality"
    )


logger.remove()
logger.add(sys.stderr, level="DEBUG")


def platipy_cli():
    """Run the PlatiPy Command Line Interface"""
    if len(sys.argv) == 1 or not sys.argv[1] in tools:
        print("")
        print("  PlatiPy CLI (Command Line Interface)")
        print("  ------------------------------------")
        print("")
        print("  Usage: platipy [tool]")
        print("")
        print("  Supply the name of the desired tool:")
        for key in tools:
            print(f"    {key}")
        print("")
        sys.exit()

    tool = sys.argv[1]
    del sys.argv[1]  # Remove the tool from the arg list so that click can run the command properly
    tools[tool]()


if __name__ == "__main__":
    platipy_cli()
