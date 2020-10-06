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


def get_lung_data(number_of_patients=3):
    """Gets some images and structures from the Head-Neck-Radiomics-HN1 (HN) dataset in TCIA.

    Keyword Arguments:
        number_of_patients {int} -- The number of patients to fetch (default: {3})

    Returns:
        dict -- dictionary of patient ids as key and path to data as value
    """

    collection = "LCTSC"
    patient_ids = get_patients_in_collection(collection)
    return fetch_data(collection, patient_ids=patient_ids[0:number_of_patients])


def get_hn_data(number_of_patients=3):
    """Gets some images and structures from the HN (Lung) dataset in TCIA.

    Keyword Arguments:
        number_of_patients {int} -- The number of patients to fetch (default: {3})

    Returns:
        dict -- dictionary of patient ids as key and path to data as value
    """

    collection = "HEAD-NECK-RADIOMICS-HN1"
    patient_ids = get_patients_in_collection(collection)
    return fetch_data(collection, patient_ids=patient_ids[0:number_of_patients])


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


def fetch_data(
    collection, patient_ids=None, output_directory=None, save_nifti=True, save_dicom=False
):
    """Fetches data from TCIA from the dataset specified

    Args:
        collection (str): The TCIA collection to fetch from
        patient_ids (list, optional): The patient IDs to fetch. If not set all patients are
                                      fetched.
        output_directory (str): The directory in which to place fetched data
        save_nifti (bool): Whether or not to save the data as Nifti in the output directory
        save_dicom (bool): Whether or not to save the data as Dicom in the output directory

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
        res = requests.get(
            series_endpoint, params={"Collection": collection, "PatientID": pid, "Modality": "CT"},
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

                if save_dicom:
                    shutil.copytree(
                        directory_to_extract_to,
                        os.path.join(patient_directory, "dicom", obj["Modality"]),
                    )

        if save_nifti:
            ct_dir = series_fetched["CT"]
            rts_file = os.path.join(
                series_fetched["RTSTRUCT"], os.listdir(series_fetched["RTSTRUCT"])[0]
            )
            convert_rtstruct(
                ct_dir,
                rts_file,
                prefix="Struct_",
                output_dir=patient_directory,
                output_img="CT.nii.gz",
            )

        # Remove the working files
        shutil.rmtree(working_dir)

    return result
