import os
import json
import zipfile
import tempfile

import shutil
import requests
import SimpleITK as sitk

from loguru import logger

api_url = "https://services.cancerimagingarchive.net/services/v4/TCIA"

collection_endpoint = f"{api_url}/query/getCollectionValues"
patient_endpoint = f"{api_url}/query/getPatient"
series_endpoint = f"{api_url}/query/getSeries"
download_series = f"{api_url}/query/getImage"


def get_bronchus_data():

    result = {}

    data_directory = os.path.join(os.path.dirname(__file__), "data", "dynamic", "lung")
    os.makedirs(data_directory, exist_ok=True)

    collection = "LCTSC"
    patient_ids = ["LCTSC-Train-S3-008", "LCTSC-Train-S1-006", "LCTSC-Train-S1-009"]

    for pid in patient_ids:

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

        for s in series:
            if s["Modality"] == "CT":

                working_dir = tempfile.mkdtemp()

                logger.debug(f"Downloading: {output_file}")

                save_path = os.path.join(working_dir, f"{pid}.zip")

                r = requests.get(
                    download_series,
                    stream=True,
                    params={"SeriesInstanceUID": s["SeriesInstanceUID"]},
                )
                with open(save_path, "wb") as fd:
                    for chunk in r.iter_content():
                        fd.write(chunk)

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
