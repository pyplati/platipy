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

from pathlib import Path

from loguru import logger

from platipy.utils import download_and_extract_zip_file

LCTSC_TEST_DATA_URL = "https://zenodo.org/record/4747795/files"
LCTSC_DICOM_ZIP = "LCTSC_DICOM_TestData.zip"
LCTSC_NIFTI_ZIP = "LCTSC_NIFTI_TestData.zip"

TCGA_HNSC_TEST_DATA_URL = "https://zenodo.org/record/5147890/files"
TCGA_HNSC_NIFTI_ZIP = "/TCGA-HNSC_NIFTI_TestData.zip"


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
