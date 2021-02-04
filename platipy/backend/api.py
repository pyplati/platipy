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

import datetime
import json
import os
import tempfile
import time
import uuid
import werkzeug

from loguru import logger

import flask_restful
from flask_restful import Api, reqparse
from flask import request, send_from_directory

from functools import wraps

from platipy.backend import app, api
from platipy.dicom.communication import DicomConnector

from .models import db, AlchemyEncoder, APIKey, Dataset, DataObject, DicomLocation
from .tasks import run_task, retrieve_task


class CustomConfig(object):
    RESTFUL_JSON = {"separators": (", ", ": "), "indent": 2, "cls": AlchemyEncoder}


def authenticate(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not getattr(func, "authenticated", True):
            return func(*args, **kwargs)

        if "API_KEY" in request.headers:
            if APIKey.query.filter_by(key=request.headers["API_KEY"]).first():
                return func(*args, **kwargs)

        flask_restful.abort(401)

    return wrapper


class Resource(flask_restful.Resource):
    method_decorators = [authenticate]  # applies to all inherited resources


class TaskStatus(Resource):
    def get(self, task_id):
        """Get the status of a task given the ID"""
        task = run_task.AsyncResult(task_id)
        if task.state == "PENDING":
            response = {
                "state": task.state,
                "current": 0,
                "total": 1,
                "status": "Pending...",
            }
        elif task.state != "FAILURE":

            if task.info:
                response = {
                    "state": task.state,
                    "current": task.info.get("current", 0),
                    "total": task.info.get("total", 1),
                    "status": task.info.get("status", ""),
                }
                if "result" in task.info:
                    response["result"] = task.info["result"]
                if "series" in task.info:
                    response["series"] = task.info["series"]
            else:
                response = {"state": task.state}

        else:
            # something went wrong in the background job
            response = {
                "state": task.state,
                "current": 1,
                "total": 1,
                "status": str(task.info),  # this is the exception raised
            }
        return response


class DicomLocationEndpoint(Resource):

    parser = reqparse.RequestParser()
    parser.add_argument("name", required=True, help="Name to identify this Dicom location")
    parser.add_argument("host", required=True, help="Dicom location host name or IP address")
    parser.add_argument("port", type=int, required=True, help="The port of the Dicom location")
    parser.add_argument("ae_title", help="AE Title of the Dicom location")

    def get(self):

        key = request.headers["API_KEY"]

        dl = DicomLocation.query.filter_by(owner_key=key).all()

        return dl

    def post(self):

        args = self.parser.parse_args()

        key = request.headers["API_KEY"]
        dl = DicomLocation(
            owner_key=key,
            name=args["name"],
            host=args["host"],
            port=args["port"],
            ae_title=args["ae_title"],
        )

        db.session.add(dl)
        db.session.commit()

        return dl


class DataObjectsEndpoint(Resource):
    def get(self):

        key = request.headers["API_KEY"]

        do = DataObject.query.filter(DataObject.dataset.has(owner_key=key)).all()

        return do


class DataObjectEndpoint(Resource):

    parser = reqparse.RequestParser()
    parser.add_argument("dataset", required=True, help="Dataset ID to add Data Object to")
    parser.add_argument(
        "type",
        choices=("DICOM", "FILE"),
        required=True,
        help="DICOM for Dicom objects to be fetched from the Dataset Dicom Location. FILE for file sent with request.",
    )
    parser.add_argument(
        "dicom_retrieve",
        choices=("MOVE", "GET", "SEND"),
        help="Used for DICOM type. The Dicom objects will be retrieved using this method.",
    )
    parser.add_argument("seriesUID")
    parser.add_argument("meta_data")
    parser.add_argument("file_name")
    parser.add_argument("file_data", type=werkzeug.datastructures.FileStorage, location="files")
    parser.add_argument("parent", help="Data Object ID to which this data object should be linked")

    def get(self, dataobject_id):

        key = request.headers["API_KEY"]

        do = DataObject.query.filter(
            DataObject.dataset.has(owner_key=key), DataObject.id == dataobject_id
        ).first()

        if do:
            return do

        return {"Error": "Data Object not found"}, 404

    def post(self):

        key = request.headers["API_KEY"]

        args = self.parser.parse_args()
        dataset_id = args["dataset"]

        # Get the dataset to which this data object should be added
        ds = Dataset.query.filter_by(owner_key=key, id=dataset_id).first()
        if not ds:
            return {"Error": "Dataset not found"}, 404

        # Get the parent dataset if one was given
        parent = None
        if args["parent"]:
            parent = DataObject.query.filter_by(dataset_id=ds.id, id=args["parent"]).first()

            if not parent:
                return {"Error": "Parent Data Object not found"}, 404

        meta_data = None
        if args["meta_data"]:
            meta_data = json.loads(args["meta_data"])

        # Create the DataObject
        do = DataObject(
            dataset=ds,
            is_input=True,
            type=args["type"],
            series_instance_uid=args["seriesUID"],
            meta_data=meta_data,
            parent=parent,
        )
        db.session.add(do)
        db.session.commit()

        if args["type"] == "DICOM":

            dicom_fetch = args["dicom_retrieve"]
            if not dicom_fetch:
                return (
                    {
                        "message": {
                            "dicom_retrieve": "Set GET, MOVE or SEND to be able to retrieve Dicom objects."
                        }
                    },
                    400,
                )

            if not args["seriesUID"]:
                return (
                    {
                        "message": {
                            "seriesUID": "SeriesUID is required to be able to retrieve DICOM objects"
                        }
                    },
                    400,
                )

            if dicom_fetch == "MOVE":

                if not ds.from_dicom_location:
                    return (
                        {
                            "message": {
                                "from_dicom_location": "Dataset From Dicom Location not set, so unable to MOVE DICOM objects"
                            }
                        },
                        400,
                    )

                # Fetch Dicom data using MOVE
                # Check whether or not we are listening for for Dicom MOVE
                listening_connector = DicomConnector(
                    host="127.0.0.1",
                    port=app.dicom_listener_port,
                    ae_title=app.dicom_listener_aetitle,
                )

                if not listening_connector.verify():

                    # Verify Dicom Location is listening
                    timeout_seconds = 20
                    time_waited = 0

                    # We are not listening, wait for 20 seconds and abort if still not listening
                    while not listening_connector.verify():
                        logger.debug(
                            "Not listening for MOVE, sleeping for 1 second and will try again"
                        )
                        time.sleep(1)
                        time_waited += 1

                        if time_waited >= timeout_seconds:
                            msg = "Listener for MOVE timeout on port: {0}".format(
                                ds.from_dicom_location.move_port
                            )
                            logger.error(msg)
                            return {"message": {"from_dicom_location": msg}}, 400

                    logger.info("Listening for MOVE OK")

                # Trigger MOVE
                logger.info(
                    "Triggering MOVE at {0} for series UID: {1}",
                    app.dicom_listener_aetitle,
                    do.series_instance_uid,
                )
                dicom_connector = DicomConnector(
                    host=ds.from_dicom_location.host,
                    port=ds.from_dicom_location.port,
                    ae_title=ds.from_dicom_location.ae_title,
                )

                dicom_verify = dicom_connector.verify()

                if dicom_verify:
                    dicom_connector.move_series(
                        do.series_instance_uid, move_aet=app.dicom_listener_aetitle
                    )
                else:
                    msg = "Unable to connect to Dicom Location: {0} {1} {2}".format(
                        ds.from_dicom_location.host,
                        ds.from_dicom_location.port,
                        ds.from_dicom_location.ae_title,
                    )
                    logger.error(msg)
                    return {"message": {"from_dicom_location": msg}}, 400

            elif dicom_fetch == "GET":

                if not ds.from_dicom_location:
                    return (
                        {
                            "message": {
                                "from_dicom_location": "Dataset From Dicom Location not set, so unable to GET DICOM objects"
                            }
                        },
                        400,
                    )

                # Fetch Dicom data using GET
                task = retrieve_task.apply_async([do.id])

            # If dicom_fetch is SEND we don't do anything here, just wait for the client
            # to send to our Dicom Listener.

        elif args["type"] == "FILE":

            if not args["file_name"]:
                return {"message": {"file_name": "Provide the file name"}}, 400

            if not args["file_data"]:
                return {"message": {"file_data": "Provide the file data"}}, 400

            # Save the file
            file_path = os.path.join(tempfile.mkdtemp(), args["file_name"])
            args["file_data"].save(file_path)
            do.is_fetched = True
            do.path = file_path

            db.session.add(do)
            db.session.commit()

        return do

    def delete(self, dataobject_id):

        key = request.headers["API_KEY"]

        do = DataObject.query.filter(
            DataObject.dataset.has(owner_key=key), DataObject.id == dataobject_id
        ).first()

        if do:
            db.session.delete(do)
            db.session.commit()

        return 200


class DataObjectDownloadEndpoint(Resource):

    parser = reqparse.RequestParser()

    def get(self, dataobject_id):
        """Returns the file for the given dataobject"""

        key = request.headers["API_KEY"]

        do = DataObject.query.filter(
            DataObject.dataset.has(owner_key=key), DataObject.id == dataobject_id
        ).first()

        if do:

            if not do.type == "FILE":
                return {"Error": "Can only download Data Objects of type FILE"}, 400

            f = do.path

            if not os.path.exists(f):
                return {"Error": "File could not be found, perhaps it has expired"}, 404

            logger.info("Downloading file: {0}".format(f))
            return send_from_directory(os.path.dirname(f), os.path.basename(f), as_attachment=True)

        return {"Error": "Data Object not found"}, 404


class DatasetsEndpoint(Resource):
    def get(self):

        key = request.headers["API_KEY"]

        ds = Dataset.query.filter_by(owner_key=key).all()

        return ds


class DatasetEndpoint(Resource):

    parser = reqparse.RequestParser()
    parser.add_argument(
        "from_dicom_location", help="ID of DicomLocation from which to retrieve DICOM data",
    )
    parser.add_argument("to_dicom_location", help="ID of DicomLocation the send output data to")
    parser.add_argument("timeout", type=int, default=24)

    def get(self, dataset_id):

        key = request.headers["API_KEY"]

        ds = Dataset.query.filter_by(owner_key=key, id=dataset_id).first()

        if ds:
            return ds

        return {"Error": "Dataset not found"}, 404

    def post(self):

        args = self.parser.parse_args()

        key = request.headers["API_KEY"]
        expiry = datetime.datetime.now() + datetime.timedelta(hours=args["timeout"])
        ds = Dataset(owner_key=key, timeout=expiry)

        if args["from_dicom_location"]:
            from_dicom_location = DicomLocation.query.filter_by(
                owner_key=key, id=args["from_dicom_location"]
            ).first()
            if not from_dicom_location:
                return {"Error": "From Dicom Location not found"}, 404
            ds.from_dicom_location = from_dicom_location

        if args["to_dicom_location"]:
            to_dicom_location = DicomLocation.query.filter_by(
                owner_key=key, id=args["to_dicom_location"]
            ).first()
            if not to_dicom_location:
                return {"Error": "To Dicom Location not found"}, 404
            ds.to_dicom_location = to_dicom_location

        db.session.add(ds)
        db.session.commit()

        return ds


class DatasetReadyEndpoint(Resource):

    parser = reqparse.RequestParser()

    def get(self, dataset_id):

        key = request.headers["API_KEY"]

        ds = Dataset.query.filter_by(owner_key=key, id=dataset_id).first()

        if ds:
            if len(ds.input_data_objects) == 0:
                return {"ready": False}, 200

            for d in ds.input_data_objects:
                if not d.is_fetched:
                    return {"ready": False}, 200

            return {"ready": True}, 200

        return {"Error": "Dataset not found"}, 404


class AlgorithmEndpoint(Resource):
    def get(self):

        result = []
        for a in app.algorithms:
            result.append({"name": a, "default_settings": app.algorithms[a].default_settings})
        return result


class TriggerEndpoint(Resource):

    parser = reqparse.RequestParser()
    parser.add_argument("algorithm", required=True, help="The name of the algorithm to trigger")
    parser.add_argument(
        "dataset", required=True, help="The ID of the dataset to pass to the algorithm"
    )
    parser.add_argument(
        "config",
        help="JSON configuration for algorithm. Default configuration will be used if not set.",
    )

    def post(self):

        args = self.parser.parse_args()

        if not args["algorithm"] in app.algorithms:
            return (
                {"Error": "No algorithm found with name: {0}".format(args["algorithm"])},
                404,
            )

        algorithm = app.algorithms[args["algorithm"]]

        config = algorithm.default_settings
        if args["config"]:
            try:
                config = json.loads(args["config"])
            except json.decoder.JSONDecodeError:
                return {"Error": "Could not parse JSON config given"}, 500

        key = request.headers["API_KEY"]
        ds = Dataset.query.filter_by(owner_key=key, id=args["dataset"]).first()
        if not ds:
            return {"Error": "Dataset not found"}, 404

        # Ensure that all input data objects in the dataset have been fetched
        if len(ds.input_data_objects) == 0:
            return {"Error": "Dataset does not contain any data objects"}, 400

        for d in ds.input_data_objects:
            if not d.is_fetched:
                return (
                    {
                        "Error": "Dataset contains Data Objects that have not been fetched yet. Wait for them to be fetched or DICOM Send them if necessary."
                    },
                    400,
                )

        # Start the algorithm task
        task = run_task.apply_async([algorithm.name, config, ds.id])

        # Return JSON data detailing where to poll for updates on the task
        return {"poll": api.url_for(TaskStatus, task_id=task.id)}


api.add_resource(TaskStatus, "/api/status/<string:task_id>")
api.add_resource(TriggerEndpoint, "/api/trigger")

api.add_resource(DatasetsEndpoint, "/api/datasets")
api.add_resource(DatasetEndpoint, "/api/dataset", "/api/dataset/<string:dataset_id>")
api.add_resource(DatasetReadyEndpoint, "/api/dataset/ready/<string:dataset_id>")

api.add_resource(DataObjectsEndpoint, "/api/dataobjects")
api.add_resource(DataObjectEndpoint, "/api/dataobject", "/api/dataobject/<string:dataobject_id>")
api.add_resource(DataObjectDownloadEndpoint, "/api/dataobject/download/<string:dataobject_id>")

api.add_resource(AlgorithmEndpoint, "/api/algorithm")

api.add_resource(
    DicomLocationEndpoint, "/api/dicomlocation", "/api/dicomlocation/<string:dicom_location_id>",
)

