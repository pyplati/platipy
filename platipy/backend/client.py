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

import time
import os
import json
from pprint import pformat

import requests

from loguru import logger


API_DICOM_LOCATION = "{0}/api/dicomlocation"
API_DATASET = "{0}/api/dataset"
API_DATASET_READY = "{0}/api/dataset/ready"
API_DATA_OBJECT = "{0}/api/dataobject"
API_TRIGGER = "{0}/api/trigger"
API_ALGORITHM = "{0}/api/algorithm"
API_DOWNLOAD_OBJECT = "{0}/api/dataobject/download"


class PlatiPyClient:
    """Client to help iteract with the framework implemented within PlatiPy."""

    def __init__(self, host, port, api_key, algorithm_name, verify=None):
        """Initialize an instance of the client. Will test to ensure that the host it reachable

        Arguments:
            host {str} -- IP or Host name of server running service
            port {int} -- Port number server is using to running service
            api_key {str} -- The API Key created for this app on the server
            algorithm_name {str} -- The name fo the algorithm to use within the service
        """

        protocol = "https"

        if verify is None:
            logger.warning("Running without SSL. Not Suitable for Production.")
            protocol = "http"
        else:
            if not os.path.exists(verify):
                raise FileNotFoundError("Verify Certificate file not found")

        self.verify = verify

        self.base_url = f"{protocol}://{host}:{port}"

        logger.info(f"Initializing client with URL: {self.base_url}")

        self.api_key = api_key
        self.algorithm_name = algorithm_name

        res = requests.get(
            API_ALGORITHM.format(self.base_url),
            headers={"API_KEY": self.api_key},
            verify=self.verify,
        )
        logger.debug(res.status_code)

    def get_dicom_location(self, name):
        """Gets the dicom location with the name supplied

        Arguments:
            name {str} -- The name of the dicom location

        Returns:
            dict -- The dicom location dictionary object, or None if it doesn't exist
        """

        res = requests.get(
            API_DICOM_LOCATION.format(self.base_url),
            headers={"API_KEY": self.api_key},
            verify=self.verify,
        )
        logger.debug(res.status_code)

        for location in res.json():
            if location["name"] == name:
                logger.debug(pformat(location))
                return location

        return None

    def add_dicom_location(self, name, host, port, ae_title=None):
        """Adds a new DICOM location.

        Arguments:
            name {str} -- The name representing this DICOM location
            host {str} -- The IP or host name of the DICOM location
            port {int} -- The port this DICOM location communicates on

        Keyword Arguments:
            ae_title {str} -- The AETitle of the DICOM location (default: {None})

        Returns:
            dict -- The DICOM location dictionary object, or None if something went wr
        """

        location = self.get_dicom_location(name)
        if location:
            logger.info(f"Location with name '{name}' already exists")
            return location

        params = {"name": name, "host": host, "port": port}

        if ae_title:
            params["ae_title"] = ae_title

        res = requests.post(
            API_DICOM_LOCATION.format(self.base_url),
            headers={"API_KEY": self.api_key},
            data=params,
            verify=self.verify,
        )
        logger.debug(res.status_code)

        if res.status_code >= 200 and res.status_code < 300:
            logger.info("Added Location")
            location = res.json()
            logger.debug(pformat(location))
            return location

        return None

    def get_dataset(self, dataset):
        """Fetches a dataset from the server

        Arguments:
            dataset {dict/int} -- The dictionary object representing the dataset or the id of the
                                  dataset

        Returns:
            dict -- The dictionary object representing the dataset
        """

        params = {"dataset": dataset}

        if isinstance(dataset, dict):
            params["dataset"] = dataset["id"]

        res = requests.get(
            "{0}/{1}".format(API_DATASET.format(self.base_url), params["dataset"]),
            headers={"API_KEY": self.api_key},
            verify=self.verify,
        )
        logger.debug(res.status_code)

        if res.status_code == 200:
            dataset = res.json()
            logger.debug(pformat(dataset))
            return dataset

        return None

    def get_dataset_ready(self, dataset):
        """Checks to see if the dataset is ready to run on the server

        Arguments:
            dataset {dict/int} -- The dictionary object representing the dataset or the id of the
                                  dataset

        Returns:
            dict -- The dictionary object representing the dataset
        """

        params = {"dataset": dataset}

        if isinstance(dataset, dict):
            params["dataset"] = dataset["id"]

        res = requests.get(
            "{0}/{1}".format(API_DATASET_READY.format(self.base_url), params["dataset"]),
            headers={"API_KEY": self.api_key},
            verify=self.verify,
        )
        logger.debug(res.status_code)

        if res.status_code == 200:
            result = res.json()
            logger.debug(pformat(result))
            return result["ready"]

        return None

    def add_dataset(self, from_dicom_location=None, to_dicom_location=None, timeout=None):
        """Adds and returns a new dataset

        Keyword Arguments:
            from_dicom_location {dict/int} -- The DICOM location to fetch objects from
                                              (default: {None})
            to_dicom_location {dict/int} -- The DICOM location to send resulting objects to
                                            (default: {None})
            timeout {int} -- How long the data within the dataset should be kept on the server
                             (default: {None})

        Returns:
            dict -- The resulting dictionary object representing the dataset or None if something
                    went wrong
        """

        params = {}

        if from_dicom_location:

            params["from_dicom_location"] = from_dicom_location

            if isinstance(from_dicom_location, dict):
                params["from_dicom_location"] = from_dicom_location["id"]

        if to_dicom_location:

            params["to_dicom_location"] = to_dicom_location

            if isinstance(to_dicom_location, dict):
                params["to_dicom_location"] = to_dicom_location["id"]

        if timeout:
            params["timeout"] = timeout

        res = requests.post(
            API_DATASET.format(self.base_url),
            headers={"API_KEY": self.api_key},
            data=params,
            verify=self.verify,
        )
        logger.debug(res.status_code)

        if res.status_code >= 200 and res.status_code < 300:
            logger.info("Added Dataset")
            dataset = res.json()
            logger.debug(pformat(dataset))
            return dataset

        return None

    def add_data_object(
        self,
        dataset,
        series_uid=None,
        parent=None,
        meta_data=None,
        dicom_retrieve=None,
        file_path=None,
    ):
        """Adds and returns an input data object to a dataset specified

        Arguments:
            dataset {dict/int} -- The dataset to which to add the data object

        Keyword Arguments:
            series_uid {str} -- The SeriesInstanceUID of the DICOM object to fetch (required for
                                DICOM objects) (default: {None})
            parent {dict/int} -- The parent object for this dataobject (default: {None})
            meta_data {dict} -- The Meta Data to attach to this data object (default: {None})
            dicom_retrieve {str} -- How the DICOM object should be fetched ['GET', 'MOVE' or
                                    'SEND'] (required for DICOM objects) (default: {None})
            file_path {str} -- Path to the file to add as the data object (Required for non-DICOM
                               objects) (default: {None})

        Returns:
            dict -- The resulting dictionary representing the data object or None if something
                    went wrong
        """

        data_object = None

        params = {"dataset": dataset}
        if isinstance(dataset, dict):
            params["dataset"] = dataset["id"]

        if parent:
            params["parent"] = parent
            if isinstance(parent, dict):
                params["parent"] = parent["id"]

        if meta_data:
            params["meta_data"] = json.dumps(meta_data)

        if series_uid or dicom_retrieve:

            if not series_uid or dicom_retrieve:
                logger.error("For Dicom, both series_uid and dicom_retrieve must be set")
                return None

            params["type"] = "DICOM"
            params["seriesUID"] = series_uid
            params["dicom_retrieve"] = dicom_retrieve

            res = requests.post(
                API_DATA_OBJECT.format(self.base_url),
                headers={"API_KEY": self.api_key},
                data=params,
                verify=self.verify,
            )
            logger.debug(res.status_code)

            if res.status_code >= 200:
                data_object = res.json()
        else:

            if not file_path:
                logger.error("For a file, provide the file_path")
                return None

            params["type"] = "FILE"
            params["file_name"] = os.path.basename(file_path)

            with open(file_path, "rb") as file_handle:

                res = requests.post(
                    API_DATA_OBJECT.format(self.base_url),
                    headers={"API_KEY": self.api_key},
                    data=params,
                    files={"file_data": file_handle},
                    verify=self.verify,
                )
                logger.debug(res.status_code)

                if res.status_code >= 200:
                    data_object = res.json()

        logger.debug(pformat(data_object))
        return data_object

    def get_default_settings(self):
        """Gets the default settings for the algorithm

        Returns:
            dict -- Dictionary object with the default settings, None if something went wrong
        """

        algorithm = None
        res = requests.get(
            API_ALGORITHM.format(self.base_url),
            headers={"API_KEY": self.api_key},
            verify=self.verify,
        )
        logger.debug(res.status_code)
        if res.status_code == 200:
            for algorithm in res.json():
                if self.algorithm_name in algorithm["name"]:
                    logger.debug(pformat(algorithm))
                    if "default_settings" in algorithm:
                        return algorithm["default_settings"]

                    # If no default settings show an error
                    logger.error("No default_settings provided by algorithm")
        return None

    def run_algorithm(self, dataset, config=None):
        """Runs the algorithm on the dataset specified

        Arguments:
            dataset {dict/int} -- The dataset on which to run the algorithm

        Yields:
            dict -- Status of the run
        """

        params = {"dataset": dataset, "algorithm": self.algorithm_name}

        if isinstance(dataset, dict):
            params["dataset"] = dataset["id"]

        if config:

            # Check that the keys of the config passed in are the exact same as the default keys
            default_settings = self.get_default_settings()

            if not set(default_settings.keys()) == set(config.keys()):
                logger.error("Config keys must be exactly those from the default_settings")
                return

            params["config"] = json.dumps(config)

        res = requests.post(
            API_TRIGGER.format(self.base_url),
            headers={"API_KEY": self.api_key},
            data=params,
            verify=self.verify,
        )
        logger.debug(res.status_code)

        if res.status_code == 200:
            # Poll the URL given to determine the progress of the task
            poll_url = "{0}{1}".format(self.base_url, res.json()["poll"])

            while True:
                res = requests.get(poll_url, headers={"API_KEY": self.api_key}, verify=self.verify)
                status = res.json()

                if (
                    not "state" in status
                    or status["state"] == "SUCCESS"
                    or status["state"] == "FAILURE"
                ):
                    break

                yield status

                time.sleep(1)
        else:
            logger.error(res.json())

        logger.info("Algorithm Processing Complete")

    def download_output_objects(self, dataset, output_path="."):
        """Downloads the output objects from the dataset specified to the output path

        Arguments:
            dataset {dict/int} -- The dataset from which the output objects should be downloaded

        Keyword Arguments:
            output_path {str} -- The directory in which the objects should be downloaded
                                 (default: {"."})
        """

        if not os.path.exists(output_path):
            logger.info("Creating directory")
            os.makedirs(output_path)

        dataset = self.get_dataset(dataset)
        if dataset:
            for data_obj in dataset["output_data_objects"]:
                url = API_DOWNLOAD_OBJECT.format(self.base_url)
                res = requests.get(
                    "{0}/{1}".format(url, data_obj["id"]),
                    headers={"API_KEY": self.api_key},
                    verify=self.verify,
                )
                logger.debug(res.status_code)
                filename = res.headers["Content-Disposition"].split("filename=")[1]

                output_file = os.path.join(output_path, filename)
                logger.info("Downloading to: {0}".format(output_file))
                open(output_file, "wb").write(res.content)
