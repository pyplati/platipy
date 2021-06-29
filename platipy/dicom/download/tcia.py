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

import json
import zipfile
import os

from pathlib import Path

import requests
from loguru import logger

from platipy.dicom.io.crawl import (
    process_dicom_directory,
)

API_URL = "https://services.cancerimagingarchive.net/services/v4/TCIA"

collection_endpoint = f"{API_URL}/query/getCollectionValues"
modalities_endpoint = f"{API_URL}/query/getModalityValues"
patient_endpoint = f"{API_URL}/query/getPatient"
series_endpoint = f"{API_URL}/query/getSeries"
series_size_endpoint = f"{API_URL}/query/getSeriesSize"
download_series_endpoint = f"{API_URL}/query/getImage"
sop_uids_endpoint = f"{API_URL}/query/getSOPInstanceUIDs"
download_image_endpoint = f"{API_URL}/query/getSingleImage"


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


def get_modalities_in_collection(collection):
    """Get a list of object modalities in a TCIA collection

    Args:
        collection (str): TCIA collection

    Returns:
        list: List of modalities available in the collection
    """

    res = requests.get(modalities_endpoint, params={"Collection": collection})
    modalities = [mod["Modality"] for mod in res.json()]

    return modalities


def get_lung_data(number_of_patients=1):
    """Gets some images and structures from the LCTSC dataset in TCIA.

    Keyword Arguments:
        number_of_patients {int} -- The number of patients to fetch (default: {1})

    Returns:
        dict -- dictionary of patient ids as key and path to data as value
    """

    collection = "LCTSC"
    patient_ids = get_patients_in_collection(collection)
    return fetch_data(
        collection, patient_ids=patient_ids[0:number_of_patients], modalities=["CT", "RTSTRUCT"]
    )


def get_hn_data(number_of_patients=1):
    """Gets some images and structures from the Head-Neck-Radiomics-HN1 dataset in TCIA.

    Keyword Arguments:
        number_of_patients {int} -- The number of patients to fetch (default: {1})

    Returns:
        dict -- dictionary of patient ids as key and path to data as value
    """

    collection = "HEAD-NECK-RADIOMICS-HN1"
    patient_ids = get_patients_in_collection(collection)
    return fetch_data(
        collection, patient_ids=patient_ids[0:number_of_patients], modalities=["CT", "RTSTRUCT"]
    )


def fetch_data(
    collection, patient_ids=None, modalities=None, nifti=True, output_directory="./tcia"
):
    """Fetches data from TCIA from the dataset specified

    Args:
        collection (str): The TCIA collection to fetch from
        patient_ids (list, optional): The patient IDs to fetch. If not set all patients are
                                      fetched
        modalities (list, optional): A list of strings definiing the modalites to fetch. Will fetch
                                     all modalities available if not specified.
        nifti (bool, optional): Whether or not to convert the fetched DICOM to Nifti. Defaults to
                                True
        output_directory (str): The directory in which to place fetched data

    Returns:
        dict: The patients and directories where their data was fetched
    """

    result = {}

    if isinstance(output_directory, str):
        output_directory = Path(output_directory)

    output_directory = output_directory.joinpath(collection)
    output_directory.mkdir(exist_ok=True, parents=True)

    modalities_available = get_modalities_in_collection(collection)
    logger.debug(f"Modalities available: {modalities_available}")

    if modalities is None:
        logger.debug("Will fetch all modalities in collection")
        modalities = modalities_available
    else:
        # Check that the user supplied modalities are all available, raise error if not
        modalities_all_available = True
        for modality in modalities:
            if not modality in modalities_available:
                modalities_all_available = False
                logger.error(f"Modality not available in collection: {modality}")

        if not modalities_all_available:
            raise ValueError("Modalities aren't all available in collection")

    if not patient_ids:
        patient_ids = get_patients_in_collection(collection)

    for pid in patient_ids:

        patient_directory = output_directory.joinpath(pid)
        dicom_directory = patient_directory.joinpath("DICOM")
        nifti_directory = patient_directory.joinpath("NIFTI")
        result[pid] = {}
        result[pid]["DICOM"] = {}

        logger.debug(f"Fetching data for Patient: {pid}")

        for modality in modalities:
            res = requests.get(
                series_endpoint,
                params={"Collection": collection, "PatientID": pid, "Modality": modality},
            )
            series = json.loads(res.text)

            if not modality in result[pid]:
                result[pid]["DICOM"][modality] = {}

            for obj in series:

                series_uid = obj["SeriesInstanceUID"]

                target_directory = dicom_directory.joinpath(series_uid)
                result[pid]["DICOM"][modality][series_uid] = target_directory
                if target_directory.exists():
                    logger.warning(
                        f"Series directory exists: {target_directory}, won't fetch data"
                    )
                    continue

                logger.debug(f"Downloading Series: {series_uid}")

                target_directory.mkdir(parents=True)

                save_path = target_directory.joinpath(f"{pid}.zip")

                response = requests.get(
                    download_series_endpoint,
                    stream=True,
                    params={"SeriesInstanceUID": obj["SeriesInstanceUID"]},
                )
                with open(save_path, "wb") as file_obj:
                    for chunk in response.iter_content():
                        file_obj.write(chunk)

                with zipfile.ZipFile(save_path, "r") as zip_ref:
                    zip_ref.extractall(target_directory)

                os.remove(save_path)

        if nifti:
            logger.info(f"Converting data for {pid} to Nifti")
            nifti_results = process_dicom_directory(
                dicom_directory, output_directory=nifti_directory
            )
            result[pid]["NIFTI"] = nifti_results[pid]

    return result
