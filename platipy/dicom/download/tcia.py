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

import os
import json
import tempfile
import zipfile
import shutil

import requests
from loguru import logger

API_URL = "https://services.cancerimagingarchive.net/services/v4/TCIA"

collection_endpoint = f"{API_URL}/query/getCollectionValues"
patient_endpoint = f"{API_URL}/query/getPatient"
series_endpoint = f"{API_URL}/query/getSeries"
download_series = f"{API_URL}/query/getImage"


def get_collections():
    """Get a list of collections available in the TCIA database

    Returns:
        list: List of collections
    """

    res = requests.get(collection_endpoint)
    collections = [collection["Collection"] for collection in res.json()]
    collections.sort()

    return collections


def get_patients_in_collection(collection):
    """Get a list of sorted patient IDs in a TCIA collection

    Args:
        collection (str): TCIA collection

    Returns:
        list: List of sorted Patient IDs in the collection
    """

    res = requests.get(patient_endpoint, params={"Collection": collection})
    patient_ids = [pat["PatientID"] for pat in res.json()]
    patient_ids.sort()

    return patient_ids


def fetch_data(collection, patient_ids=None, output_directory=None):
    """Fetches data from TCIA from the dataset specified

    Args:
        collection (str): The TCIA collection to fetch from
        patient_ids (list, optional): The patient IDs to fetch. If not set all patients are
                                      fetched.
        output_directory (str): The directory in which to place fetched data

    Returns:
        dict: The patients and directories where their data was fetched
    """

    result = {}

    if not output_directory:
        output_directory = os.path.join(os.path.dirname(__file__), "data", collection)

    os.makedirs(output_directory, exist_ok=True)

    modalities = ["CT", "RTSTRUCT"]

    if not patient_ids:
        patient_ids = get_patients_in_collection(collection)

    for pid in patient_ids:

        patient_directory = os.path.join(output_directory, pid)
        result[pid] = patient_directory
        if os.path.exists(patient_directory):
            logger.debug(f"Path exists: {patient_directory}, won't fetch data")
            continue

        logger.debug(f"Fetching data for Patient: {pid}")

        for modality in modalities:
            res = requests.get(
                series_endpoint,
                params={"Collection": collection, "PatientID": pid, "Modality": modality},
            )
            series = json.loads(res.text)
            series_fetched = {}
            working_dir = tempfile.mkdtemp()

            for obj in series:

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

                shutil.copytree(
                    directory_to_extract_to,
                    os.path.join(patient_directory, "dicom", obj["Modality"]),
                )

            # Remove the working files
            shutil.rmtree(working_dir)

    return result
