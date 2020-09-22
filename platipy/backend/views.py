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

from loguru import logger
import psutil

from flask import Flask, request, render_template, jsonify

from platipy.backend import app, celery, log_file_path
from platipy.dicom.communication import DicomConnector
from .models import db, APIKey


@app.route("/endpoint/add", methods=["GET"])
def add_endpoint():

    return render_template("endpoint_add.html", data=app.data)


@app.route("/log", methods=["GET"])
def fetch_log():

    log = []
    with open(log_file_path) as f:

        for line in f:
            log.append(line.replace("\n", ""))

    return jsonify({"log": log})


@app.route("/endpoint/<id>", methods=["GET", "POST"])
def view_endpoint(id):

    endpoint = None
    for e in app.data["endpoints"]:
        if e["id"] == int(id):
            endpoint = e

    status = ""
    # Check if the last is still running
    if endpoint["endpointType"] == "listener":
        if "task_id" in endpoint:
            task = listen_task.AsyncResult(endpoint["task_id"])
            status = task.info.get("status", "")
            if "Error" in status:
                kill_task(endpoint["task_id"])

    return render_template(
        "endpoint_view.html",
        data=app.data,
        endpoint=endpoint,
        status=status,
        format_settings=lambda x: json.dumps(x, indent=4),
    )


@app.route("/status", methods=["GET"])
def fetch_status():

    celery_running = False
    if celery.control.inspect().active():
        celery_running = True
    status_context = {"celery": celery_running}
    status_context["algorithms"] = []
    for a in app.algorithms:
        algorithm = app.algorithms[a]
        status_context["algorithms"].append(
            {"name": algorithm.name, "default_settings": algorithm.default_settings}
        )

    dicom_connector = DicomConnector(
        port=app.dicom_listener_port, ae_title=app.dicom_listener_aetitle
    )
    dicom_listening = False
    if dicom_connector.verify():
        dicom_listening = True
    status_context["dicom_listener"] = {
        "port": app.dicom_listener_port,
        "aetitle": app.dicom_listener_aetitle,
        "listening": dicom_listening,
    }

    status_context["ram_usage"] = psutil.virtual_memory()._asdict()
    status_context["disk_usage"] = psutil.disk_usage("/")._asdict()
    status_context["cpu_usage"] = psutil.cpu_percent()

    status_context["applications"] = []
    for ak in APIKey.query.all():
        status_context["applications"].append({"name": ak.name, "key": ak.key})

    return jsonify(status_context)


@app.route("/")
def dashboard():
    """Entry point to the dashboard of the application"""

    return render_template("dashboard.html", data={})
