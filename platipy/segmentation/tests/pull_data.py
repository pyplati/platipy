"""
Functions to pull data for tests
"""

import os
import json
import zipfile
import tempfile

import shutil
import requests

from loguru import logger

from platipy.dicom.rtstruct_to_nifti.convert import convert_rtstruct

API_URL = "https://services.cancerimagingarchive.net/services/v4/TCIA"

collection_endpoint = f"{API_URL}/query/getCollectionValues"
patient_endpoint = f"{API_URL}/query/getPatient"
series_endpoint = f"{API_URL}/query/getSeries"
download_series = f"{API_URL}/query/getImage"

datasets = {
    "lung": {
        "collection": "LCTSC",
        "modalities": ["CT", "RTSTRUCT"],
        "patient_ids": [
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
        ],
    }
}


def get_lung_data(number_of_patients=3):
    """Gets some images and structures from the LCTSC (Lung) dataset in TCIA.

    Keyword Arguments:
        number_of_patients {int} -- The number of patients to fetch (default: {3})

    Returns:
        dict -- dictionary of patient ids as key and path to data as value
    """

    return fetch_data("lung", datasets["lung"]["patient_ids"][0:number_of_patients])


def fetch_data(dataset, patient_ids=None):
    """Fetches data from TCIA from the dataset specified

    Args:
        dataset (str): The platipy id of the dataset to fetch from (see datasets)
        patient_ids (list, optional): The patient IDs to fetch. If not set all patients are
                                      fetched.

    Returns:
        dict: The patients and directories where their data was fetched to
    """

    result = {}

    data_directory = os.path.join(os.path.dirname(__file__), "data", "dynamic", dataset)

    collection = datasets[dataset]["collection"]
    modalities = datasets[dataset]["modalities"]

    if not patient_ids:
        patient_ids = datasets[dataset]["patient_ids"]

    for pid in patient_ids:

        output_directory = os.path.join(data_directory, pid)
        result[pid] = output_directory
        if os.path.exists(output_directory):
            logger.debug(f"Path exists: {output_directory}, won't fetch data")
            continue

        logger.debug(f"Fetching data for Patient: {pid}")
        res = requests.get(
            series_endpoint, params={"Collection": collection, "PatientID": pid, "Modality": "CT"}
        )
        series = json.loads(res.text)
        series_fetched = {}
        working_dir = tempfile.mkdtemp()

        for obj in series:

            if obj["Modality"] in modalities:

                save_path = os.path.join(working_dir, f"{pid}.zip")

                response = requests.get(
                    download_series,
                    stream=True,
                    params={"SeriesInstanceUID": obj["SeriesInstanceUID"]},
                )
                with open(save_path, "wb") as file_obj:
                    for chunk in response.iter_content():
                        file_obj.write(chunk)

                directory_to_extract_to = os.path.join(working_dir, pid, obj["Modality"])
                with zipfile.ZipFile(save_path, "r") as zip_ref:
                    zip_ref.extractall(directory_to_extract_to)

                series_fetched[obj["Modality"]] = directory_to_extract_to

        ct_dir = series_fetched["CT"]
        rts_file = os.path.join(
            series_fetched["RTSTRUCT"], os.listdir(series_fetched["RTSTRUCT"])[0]
        )
        convert_rtstruct(
            ct_dir,
            rts_file,
            prefix="Struct_",
            output_dir=output_directory,
            output_img="CT.nii.gz",
        )

        # Remove the working files
        shutil.rmtree(working_dir)

    return result
