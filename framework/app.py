from flask import Flask
from celery import current_app
from celery.bin import worker
from multiprocessing import Process
from loguru import logger
import json
import os
import uuid

# TODO configure log file properly
logger.add("logfile.log")


class Algorithm():

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
    dicom_listener_port = 7777
    dicom_listener_aetitle = "IMPIT_SERVICE"

    api = None # Holds reference to api for extensibility

    def __init__(self, name):

        super().__init__(name)

    def register(self, name, default_settings=None):

        def decorator(f):
            self.algorithms.update(
                {name: Algorithm(name, f, default_settings)})
            return f

        return decorator

    def run_celery(self):

        if self.celery_started:
            return

        application = current_app._get_current_object()

        celery_worker = worker.worker(app=application)

        options = {
            'broker': web_app.config['CELERY_BROKER_URL'],
            'loglevel': 'INFO',
            'traceback': True,
        }

        celery_worker.run(**options)

    def run(self, host=None, port=None, debug=None,
            dicom_listener_port=7777,
            dicom_listener_aetitle="IMPIT_SERVICE",
            load_dotenv=True, **options):

        logger.info('Starting APP!')

        p = Process(target=self.run_celery)
        p.start()
        self.celery_started = True

        self.dicom_listener_port = dicom_listener_port
        self.dicom_listener_aetitle = dicom_listener_aetitle
        logger.info('Starting Dicom Listener on port: {0} with AE Title: {1}',
            dicom_listener_port,
            dicom_listener_aetitle)
        from .tasks import listen_task
        listen_task.apply_async([
            dicom_listener_port,
            dicom_listener_aetitle
        ])

        super().run(host=host, port=port, debug=debug,
                    load_dotenv=load_dotenv, use_reloader=False, **options)

        p.join()

    def test_client(self, use_cookies=True, **kwargs):

        self.init_app()

        return super().test_client(use_cookies=use_cookies, **kwargs)


web_app = FlaskApp(__name__)
web_app.config['SECRET_KEY'] = uuid.uuid4().hex

import impit.framework.api
import impit.framework.views
import impit.framework.tasks
import impit.framework.models

# Import DataObject for easy import from algorithm
from impit.framework.models import DataObject