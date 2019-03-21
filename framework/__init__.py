from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from celery import Celery
from celery import current_app
from celery.bin import worker
from celery.task.control import revoke

from .application import FlaskApp

import os
import uuid

# Create Flask app
app = FlaskApp(__name__)
app.config['SECRET_KEY'] = uuid.uuid4().hex

# Configure SQL Alchemy

# # TODO place this in the working directory
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///{0}/{1}.db'.format(os.getcwd(), os.path.basename(os.getcwd()))
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configure Celery

# TODO Should be in a configuration file
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['result_backend'] = 'redis://localhost:6379/0'
celery = Celery(__name__, broker='redis://localhost:6379/0')
celery.conf.update(app.config)

import impit.framework.tasks
import impit.framework.api
import impit.framework.views
from impit.framework.models import DataObject