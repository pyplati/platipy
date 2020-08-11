from flask import Flask
from celery import current_app
from celery.bin import worker
from celery.bin import beat
from multiprocessing import Process
from loguru import logger
import json
import os
import uuid
import tempfile


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

        pc = Process(target=self.run_celery)
        pc.start()
        self.celery_started = True

        pb = Process(target=self.run_beat)
        pb.start()
        self.beat_started = True

        self.dicom_listener_port = dicom_listener_port
        self.dicom_listener_aetitle = dicom_listener_aetitle

        from .tasks import listen_task

        listen_task.apply_async([dicom_listener_port, dicom_listener_aetitle])

        super().run(
            host=host,
            port=port,
            debug=debug,
            load_dotenv=load_dotenv,
            use_reloader=False,
            **options
        )

        pc.join()
        pb.join()

    # def test_client(self, use_cookies=True, **kwargs):

    #     self.init_app()

    #     return super().test_client(use_cookies=use_cookies, **kwargs)
