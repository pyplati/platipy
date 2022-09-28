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

import logging
import sys
import os
import uuid

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_restful import Api
from celery import Celery
import celery.signals

from platipy.backend.application import FlaskApp

env_work = os.getcwd()
if "WORK" in os.environ:
    env_work = os.environ["WORK"]
log_file_path = os.path.join(env_work, "service.log")


def configure_logging():
    logger = logging.getLogger()

    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=100 * 1024 * 1024,  # Max 100 MB per log file before rotating
        backupCount=100,  # Keep up to 100 log files in history
    )
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)


@celery.signals.setup_logging.connect
def on_celery_setup_logging(**kwargs):
    configure_logging()


configure_logging()

# Create Flask app
app = FlaskApp(__name__)
app.config["SECRET_KEY"] = uuid.uuid4().hex

# Configure SQL Alchemy
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{env_work}/{os.path.basename(os.getcwd())}.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Configure Celery
REDIS_HOST = "localhost"
if "REDIS_HOST" in os.environ:
    REDIS_HOST = os.environ["REDIS_HOST"]

REDIS_PORT = 6379
if "REDIS_PORT" in os.environ:
    REDIS_PORT = os.environ["REDIS_PORT"]

app.config["CELERY_BROKER_URL"] = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
app.config["result_backend"] = app.config["CELERY_BROKER_URL"]
app.config["worker_max_memory_per_child"] = 32 * 1000 * 1000  # 16GB in KB
celery = Celery(__name__, broker=app.config["CELERY_BROKER_URL"])
celery.conf.update(app.config)

# Configure API
api = Api(app)
app.config.from_object("platipy.backend.api.CustomConfig")

import platipy.backend.views
import platipy.backend.api
import platipy.backend.tasks
from platipy.backend.models import DataObject
