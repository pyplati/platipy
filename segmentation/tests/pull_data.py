"""
Functions to pull data for tests
"""

import os
import json
import zipfile
import tempfile

import shutil
import requests
import SimpleITK as sitk

from loguru import logger

API_URL = "https://services.cancerimagingarchive.net/services/v4/TCIA"

collection_endpoint = f"{API_URL}/query/getCollectionValues"
patient_endpoint = f"{API_URL}/query/getPatient"
series_endpoint = f"{API_URL}/query/getSeries"
download_series = f"{API_URL}/query/getImage"


def get_bronchus_data(number_of_images=3):
    """Finds some images from the LCTSC (Lung) dataset in TCIA. Suitable for bronchus segmentation.

    Keyword Arguments:
        number_of_images {int} -- The number of images to fetch (default: {3})

    Returns:
        dict -- dictionary of patient ids as key and path to image as value
    """

    result = {}

    data_directory = os.path.join(os.path.dirname(__file__), "data", "dynamic", "lung")
    os.makedirs(data_directory, exist_ok=True)

    collection = "LCTSC"

    patient_ids = [
        "LCTSC-Train-S1-001",
        "LCTSC-Train-S1-002",
        "LCTSC-Train-S1-003",
        "LCTSC-Train-S1-004",
        "LCTSC-Train-S1-005",
        "LCTSC-Train-S1-006",
        "LCTSC-Train-S1-007",
        "LCTSC-Train-S1-008",
        "LCTSC-Train-S1-009",
        "LCTSC-Train-S1-010",
        "LCTSC-Train-S1-011",
        "LCTSC-Train-S1-012",
        "LCTSC-Train-S2-001",
        "LCTSC-Train-S2-002",
        "LCTSC-Train-S2-003",
        "LCTSC-Train-S2-004",
        "LCTSC-Train-S2-005",
        "LCTSC-Train-S2-006",
        "LCTSC-Train-S2-007",
        "LCTSC-Train-S2-008",
        "LCTSC-Train-S2-009",
        "LCTSC-Train-S2-010",
        "LCTSC-Train-S2-011",
        "LCTSC-Train-S2-012",
        "LCTSC-Train-S3-001",
        "LCTSC-Train-S3-002",
        "LCTSC-Train-S3-003",
        "LCTSC-Train-S3-004",
        "LCTSC-Train-S3-005",
        "LCTSC-Train-S3-006",
        "LCTSC-Train-S3-007",
        "LCTSC-Train-S3-008",
        "LCTSC-Train-S3-009",
        "LCTSC-Train-S3-010",
        "LCTSC-Train-S3-011",
        "LCTSC-Train-S3-012",
        "LCTSC-Test-S1-101",
        "LCTSC-Test-S1-102",
        "LCTSC-Test-S1-103",
        "LCTSC-Test-S1-104",
        "LCTSC-Test-S1-201",
        "LCTSC-Test-S1-202",
        "LCTSC-Test-S1-203",
        "LCTSC-Test-S1-204",
        "LCTSC-Test-S2-101",
        "LCTSC-Test-S2-102",
        "LCTSC-Test-S2-103",
        "LCTSC-Test-S2-104",
        "LCTSC-Test-S2-201",
        "LCTSC-Test-S2-202",
        "LCTSC-Test-S2-203",
        "LCTSC-Test-S2-204",
        "LCTSC-Test-S3-101",
        "LCTSC-Test-S3-102",
        "LCTSC-Test-S3-103",
        "LCTSC-Test-S3-104",
        "LCTSC-Test-S3-201",
        "LCTSC-Test-S3-202",
        "LCTSC-Test-S3-203",
        "LCTSC-Test-S3-204",
    ]

    for pid in patient_ids:

        if len(result.keys()) == number_of_images:
            break

        output_file = os.path.join(data_directory, f"CT_{pid}.nii.gz")
        result[pid] = output_file
        if os.path.exists(output_file):
            logger.debug(f"File exists: {output_file}")
            continue

        logger.debug(f"Fetching Series for Patient: {pid}")
        res = requests.get(
            series_endpoint, params={"Collection": collection, "PatientID": pid, "Modality": "CT"}
        )
        series = json.loads(res.text)

        for img_series in series:

            # Need to check that modality is CT, not sure why
            if img_series["Modality"] == "CT":

                working_dir = tempfile.mkdtemp()

                logger.debug(f"Downloading: {output_file}")

                save_path = os.path.join(working_dir, f"{pid}.zip")

                response = requests.get(
                    download_series,
                    stream=True,
                    params={"SeriesInstanceUID": img_series["SeriesInstanceUID"]},
                )
                with open(save_path, "wb") as file_obj:
                    for chunk in response.iter_content():
                        file_obj.write(chunk)

                directory_to_extract_to = os.path.join(working_dir, pid)
                with zipfile.ZipFile(save_path, "r") as zip_ref:
                    zip_ref.extractall(directory_to_extract_to)

                series_files = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(
                    directory_to_extract_to
                )

                img = sitk.ReadImage(series_files)

                sitk.WriteImage(img, output_file)

                # Remove the working files
                shutil.rmtree(working_dir)

    return result
