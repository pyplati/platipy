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
import tempfile

from flask import Flask
from celery import current_app
from celery.bin import worker
from celery.bin import beat
from multiprocessing import Process
from loguru import logger
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

    def run_celery(self):

        if self.celery_started:
            return

        application = current_app._get_current_object()

        celery_worker = worker.worker(app=application)

        options = {
            "broker": self.config["CELERY_BROKER_URL"],
            "loglevel": "INFO",
            "traceback": True,
        }

        celery_worker.run(**options)

    def run_beat(self):

        if self.beat_started:
            return

        application = current_app._get_current_object()

        celery_beat = beat.beat(app=application)

        options = {
            "broker": self.config["CELERY_BROKER_URL"],
            "loglevel": "INFO",
            "traceback": True,
            "beat": True,
            "schedule": os.path.join(str(tempfile.mkdtemp()), "celery-beat-schedule"),
        }

        celery_beat.run(**options)

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

        process_celery = Process(target=self.run_celery)
        process_celery.start()
        self.celery_started = True

        process_beat = Process(target=self.run_beat)
        process_beat.start()
        self.beat_started = True

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

        process_celery.join()
        process_beat.join()

    def run_dicom_listener(self, listen_port, listen_ae_title):
        """
        Background task that listens at a specific port for incoming dicom series
        """

        from .models import Dataset, DataObject
        from . import db

        logger.info(
            "Starting Dicom Listener on port: {0} with AE Title: {1}",
            listen_port,
            listen_ae_title,
        )

        def series_recieved(dicom_path):
            logger.info("Series Recieved at path: {0}".format(dicom_path))

            # Get the SeriesUID
            series_uid = None
            for f in os.listdir(dicom_path):
                f = os.path.join(dicom_path, f)

                try:
                    d = pydicom.read_file(f)
                    series_uid = d.SeriesInstanceUID
                except Exception as e:
                    logger.debug("No Series UID in: {0}".format(f))
                    logger.debug(e)

            if series_uid:
                logger.info("Image Series UID: {0}".format(series_uid))
            else:
                logger.error("Series UID could not be determined... Stopping")
                return

            # Find the data objects with the given series UID and update them
            dos = DataObject.query.filter_by(series_instance_uid=series_uid).all()

            if len(dos) == 0:
                logger.error(
                    "No Data Object found with Series UID: {0} ... Stopping".format(series_uid)
                )
                return

            for do in dos:

                do.is_fetched = True
                do.path = dicom_path
                db.session.commit()

        try:
            dicom_listener = DicomListener(
                port=listen_port, ae_title=listen_ae_title, on_released_callback=series_recieved
            )

            dicom_listener.start()

        except Exception as e:
            logger.error("Listener Error: " + str(e))
