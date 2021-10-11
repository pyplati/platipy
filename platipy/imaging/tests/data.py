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

import zipfile
import tempfile
import urllib.request
from pathlib import Path

from loguru import logger


LCTSC_TEST_DATA_URL = "https://zenodo.org/record/4747795/files"
LCTSC_DICOM_ZIP = "LCTSC_DICOM_TestData.zip"
LCTSC_NIFTI_ZIP = "LCTSC_NIFTI_TestData.zip"

TCGA_HNSC_TEST_DATA_URL = "https://zenodo.org/record/5147890/files"
TCGA_HNSC_NIFTI_ZIP = "/TCGA-HNSC_NIFTI_TestData.zip"


def download_and_extract_zip_file(zip_url, output_directory):

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir).joinpath("tmp.zip")

        with urllib.request.urlopen(zip_url) as dl_file:
            with open(temp_file, "wb") as out_file:
                out_file.write(dl_file.read())

        with zipfile.ZipFile(temp_file, "r") as zip_ref:
            zip_ref.extractall(output_directory)


def get_lung_dicom(output_directory="./data/dicom"):

    output_directory = Path(output_directory)

    if output_directory.exists():
        logger.debug(f"Output directory exists, stopping. {output_directory}")
        return output_directory

    zip_url = f"{LCTSC_TEST_DATA_URL}/{LCTSC_DICOM_ZIP}?download=1"
    download_and_extract_zip_file(zip_url, output_directory)

    return output_directory


def get_lung_nifti(output_directory="./data/nifti/lung"):

    output_directory = Path(output_directory)

    if output_directory.exists():
        logger.debug(f"Output directory exists, stopping. {output_directory}")
        return output_directory

    zip_url = f"{LCTSC_TEST_DATA_URL}/{LCTSC_NIFTI_ZIP}?download=1"
    download_and_extract_zip_file(zip_url, output_directory)

    return output_directory


def get_hn_nifti(output_directory="./data/nifti/hn"):

    output_directory = Path(output_directory)

    if output_directory.exists():
        logger.debug(f"Output directory exists, stopping. {output_directory}")
        return output_directory

    zip_url = f"{TCGA_HNSC_TEST_DATA_URL}/{TCGA_HNSC_NIFTI_ZIP}?download=1"
    download_and_extract_zip_file(zip_url, output_directory)

    return output_directory
