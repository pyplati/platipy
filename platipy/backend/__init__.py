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
import uuid

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_restful import Api
from celery import Celery

from platipy.backend.application import FlaskApp

from loguru import logger

env_work = os.getcwd()
if "WORK" in os.environ:
    env_work = os.environ["WORK"]

# Configure Log file location
log_file_path = os.path.join(env_work, "service.log")
logger.add(log_file_path, rotation="1 day")

# Create Flask app
app = FlaskApp(__name__)
app.config["SECRET_KEY"] = uuid.uuid4().hex

# Configure SQL Alchemy
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///{0}/{1}.db".format(
    env_work, os.path.basename(os.getcwd())
)
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
