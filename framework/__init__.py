import os
import uuid

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_restful import Api
from celery import Celery

from .application import FlaskApp

from loguru import logger

env_work = os.getcwd()
if 'WORK' in os.environ:
    env_work = os.environ['WORK']

# Configure Log file location
log_file_path = os.path.join(env_work, "service.log")
logger.add(log_file_path, rotation="1 day")

# Create Flask app
app = FlaskApp(__name__)
app.config['SECRET_KEY'] = uuid.uuid4().hex

# Configure SQL Alchemy

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///{0}/{1}.db'.format(
    env_work, os.path.basename(os.getcwd()))
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configure Celery

# TODO Should be in a configuration file
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['result_backend'] = 'redis://localhost:6379/0'
app.config['worker_max_memory_per_child'] = 5 * 1000 * 1000 # 5GB in KB
celery = Celery(__name__, broker='redis://localhost:6379/0')
celery.conf.update(app.config)

# Configure API
api = Api(app)
app.config.from_object('impit.framework.api.CustomConfig')

import impit.framework.views
import impit.framework.api
import impit.framework.tasks
from impit.framework.models import DataObject
