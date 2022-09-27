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
import os

from flask import Flask
import logging

logger = logging.getLogger(__name__)
import pydicom

from pymedphys._dicom.connect.listen import DicomListener


class Algorithm:
    def __init__(self, name, function, default_settings):
        self.name = name
        self.function = function
        self.default_settings = default_settings

    def settings_to_json(self):
        return json.dumps(self.default_settings, indent=4)


class FlaskApp(Flask):
    """
    Custom Flask App
    """

    algorithms = {}
    celery_started = False
    beat_started = False
    dicom_listener_port = 7777
    dicom_listener_aetitle = "PLATIPY_SERVICE"

    api = None  # Holds reference to api for extensibility

    def register(self, name, default_settings=None):
        def decorator(f):
            self.algorithms.update({name: Algorithm(name, f, default_settings)})
            return f

        return decorator

    def run(
        self,
        host=None,
        port=None,
        debug=None,
        dicom_listener_port=7777,
        dicom_listener_aetitle="PLATIPY_SERVICE",
        load_dotenv=True,
        **options
    ):

        logger.info("Starting APP!")

        self.dicom_listener_port = dicom_listener_port
        self.dicom_listener_aetitle = dicom_listener_aetitle

        self.run_dicom_listener(dicom_listener_port, dicom_listener_aetitle)

        super().run(
            host=host,
            port=port,
            debug=debug,
            load_dotenv=load_dotenv,
            use_reloader=False,
            **options
        )

    def run_dicom_listener(self, listen_port=None, listen_ae_title=None):
        """
        Background task that listens at a specific port for incoming dicom series
        """

        from .models import Dataset, DataObject
        from . import db

        if listen_port is None:
            listen_port = self.dicom_listener_port

        if listen_ae_title is None:
            listen_ae_title = self.dicom_listener_aetitle

        logger.info(
            "Starting Dicom Listener on port: %s with AE Title: %s",
            listen_port,
            listen_ae_title,
        )

        def series_recieved(dicom_path):
            logger.info("Series Recieved at path: %s", dicom_path)

            # Get the SeriesUID
            series_uid = None
            for f in os.listdir(dicom_path):
                f = os.path.join(dicom_path, f)

                try:
                    d = pydicom.read_file(f)
                    series_uid = d.SeriesInstanceUID
                except Exception as e:
                    logger.debug("No Series UID in: %s", f)
                    logger.debug(e)

            if series_uid:
                logger.info("Image Series UID: %s", series_uid)
            else:
                logger.error("Series UID could not be determined... Stopping")
                return

            # Find the data objects with the given series UID and update them
            dos = DataObject.query.filter_by(series_instance_uid=series_uid).all()

            if len(dos) == 0:
                logger.error("No Data Object found with Series UID: %s ... Stopping", series_uid)
                return

            for do in dos:

                do.is_fetched = True
                do.path = str(dicom_path)
                db.session.commit()

        try:
            dicom_listener = DicomListener(
                host="0.0.0.0",
                port=listen_port,
                ae_title=listen_ae_title,
                on_released_callback=series_recieved,
            )

            dicom_listener.start()

        except Exception as e:
            logger.error("Listener Error: %s", e)
