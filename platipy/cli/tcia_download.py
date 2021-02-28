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

import click
from loguru import logger

from platipy.dicom.download.tcia import (
    get_collections,
    get_patients_in_collection,
    get_modalities_in_collection,
    fetch_data,
)


logger.remove()
logger.add(sys.stderr, level="DEBUG")


@click.command()
@click.argument(
    "collection",
    type=str,
)
@click.option(
    "--patient",
    "-p",
    "patients",
    required=False,
    multiple=True,
    type=str,
    help="ID of patient to download (if not specified all patients will be downloaded,"
    "can be specified multiple times)",
)
@click.option(
    "--modality",
    "-m",
    "modalities",
    required=False,
    multiple=True,
    type=str,
    help="Modality to download (if not specified all modalities will be downladed, can be "
    "multiple times).",
)
@click.option(
    "--nifti/--no-nifti",
    default=True,
    help="Flag whether to convert data to NIFTI or not",
)
@click.option(
    "--output", "-o", required=False, type=click.Path(), help="Path to directory to store output"
)
def click_command(collection, patients, modalities, nifti, output):
    """
    This tool allows you to download data directly from The Cancer Imaging Archive (TCIA).

    Data available on TCIA is listed here: https://www.cancerimagingarchive.net/collections/
    """

    collections = get_collections()
    if not collection in collections:
        logger.error(f"Collection '{collection}' not found on TCIA")
        return

    logger.info(f"Downloading from collection '{collection}'")

    tcia_patients = get_patients_in_collection(collection)
    patients_ok = True
    for patient in patients:
        if not patient in tcia_patients:
            logger.error(f"Patient '{patient}' not found in collection")
            patients_ok = False

    if not patients_ok:
        return

    if len(patients) == 0:
        patients = tcia_patients

    logger.info(f"Downloading patients: {patients}")

    tcia_modalities = get_modalities_in_collection(collection)
    modalities_ok = True
    for modality in modalities:
        if not modality in tcia_modalities:
            logger.error(f"Modality '{modality}' not available in collection")
            modalities_ok = False

    if not modalities_ok:
        return

    if len(modalities) == 0:
        modalities = tcia_modalities

    logger.info(f"Downloading modalities: {modalities}")
    logger.info(f"Convert DICOM to NIFTI: {nifti}")

    if output is None:
        output = "./tcia"
    logger.info(f"Downloading data to: {output}")

    fetch_data(
        collection,
        patient_ids=patients,
        modalities=modalities,
        nifti=nifti,
        output_directory=output,
    )


if __name__ == "__main__":

    click_command()  # pylint: disable=no-value-for-parameter
