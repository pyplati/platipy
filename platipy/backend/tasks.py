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

from celery.schedules import crontab

from loguru import logger
import tempfile
import pydicom
import os
import datetime
import shutil
import time


from platipy.backend import celery, db, app
from platipy.dicom.communication import DicomConnector

from .models import Dataset, DataObject

celery.conf.beat_schedule = {
    # Executes every hour
    "clean-up-every-hour": {
        "task": "platipy.backend.tasks.clean_up_task",
        "schedule": crontab(minute=0),
        "args": (),
    },
}

celery.conf.timezone = "UTC"


@celery.task(bind=True)
def clean_up_task(task):
    """
    Deletes DataObjects from expired Datasets
    """

    logger.info("Running Clean Up Task")

    datasets = Dataset.query.all()

    now = datetime.datetime.now()

    num_data_objs_removed = 0

    for ds in datasets:

        if ds.timeout < now:

            data_objects = ds.input_data_objects + ds.output_data_objects

            for do in data_objects:

                try:
                    if do.path:
                        if os.path.isdir(do.path):
                            logger.debug("Removing Directory: ", do.path)
                            shutil.rmtree(do.path)
                        elif os.path.isfile(do.path):
                            logger.debug("Removing File: ", do.path)
                            os.remove(do.path)
                        else:
                            logger.debug(
                                "Data missing, must have already been deleted: ",
                                do.path,
                            )

                        num_data_objs_removed += 1

                        do.is_fetched = False
                        do.path = None
                        db.session.commit()

                except Exception as e:
                    logger.warning(
                        "Exception occured when removing DataObject: " + str(do)
                    )

    logger.info(
        "Clean Up Task Complete: Removed {0} DataObjects", num_data_objs_removed
    )


@celery.task(bind=True)
def retrieve_task(task, data_object_id):
    """
    Fetch a Dicom Object from the dicom location using the retrieve_type GET or MOVE
    """

    do = DataObject.query.filter_by(id=data_object_id).first()

    dicom_connector = DicomConnector(
        host=do.dataset.from_dicom_location.host,
        port=do.dataset.from_dicom_location.port,
        ae_title=do.dataset.from_dicom_location.ae_title,
    )
    dicom_verify = dicom_connector.verify()

    if not dicom_verify:
        logger.error(
            "Unable to connect to Dicom Location: {0} {1} {2}".format(
                do.dataset.from_dicom_location.host,
                do.dataset.from_dicom_location.port,
                do.dataset.from_dicom_location.ae_title,
            )
        )
        return

    dicom_path = dicom_connector.download_series(do.series_instance_uid)

    do.is_fetched = True
    do.path = dicom_path
    db.session.commit()


@celery.task(bind=True)
def move_task(task, endpoint, seriesUIDs, host, port, ae_title):
    """
    Background task that triggers the Dicom MOVE operation at the given endpoint
    for the given seriesUIDs
    """

    # For each series UID supplied, fetch the image series and run the algorithm
    total = len(seriesUIDs)
    count = 0

    dicom_connector = DicomConnector(host=host, port=port, ae_title=ae_title)

    task.update_state(
        state="PROGRESS",
        meta={"current": count, "total": total, "status": "Verifying dicom location"},
    )

    dicom_verify = dicom_connector.verify()

    if dicom_verify == None:
        return {
            "current": 100,
            "total": 100,
            "status": "Unable to connect to dicom location",
        }

    for suid in seriesUIDs:

        task.update_state(
            state="PROGRESS",
            meta={
                "current": count,
                "total": total,
                "status": "Moving series for UID: {0}".format(suid),
            },
        )

        logger.info("Moving Series with UID: {0}".format(suid))
        dicom_connector.move_series(suid)

        count = count + 1

    task.update_state(
        state="SUCCESS",
        meta={"current": total, "total": total, "status": "Move Complete"},
    )


@celery.task(bind=True)
def run_task(task, algorithm_name, config, dataset_id):

    task_id = task.request.id

    start = time.time()

    # Commit to refresh session
    db.session.commit()

    algorithm = app.algorithms[algorithm_name]

    if not config:
        config = algorithm.default_settings

    ds = Dataset.query.filter_by(id=dataset_id).first()
    input_objects = ds.input_data_objects

    logger.info("Will run algorithm: " + algorithm_name)
    logger.info("Using settings: " + str(config))
    logger.info("Number of data objects in dataset: " + str(len(input_objects)))

    state_details = {
        "current": 0,
        "total": len(input_objects),
        "status": "Running Algorithm: {0}".format(algorithm_name),
    }

    task.update_state(state="RUNNING", meta=state_details)

    if config is None:
        output_data_objects = algorithm.function(input_objects, tempfile.mkdtemp())
    else:
        output_data_objects = algorithm.function(
            input_objects, tempfile.mkdtemp(), config
        )

    if not output_data_objects:
        logger.warning(
            "Algorithm ({0}) did not return any output objects".format(algorithm_name)
        )

    # Save the data objects
    for do in output_data_objects:
        do.dataset_id = ds.id
        db.session.add(do)
        db.session.commit()

        if do.type == "DICOM":
            if ds.to_dicom_location:

                logger.info("Sending to Dicom To Location")
                dicom_connector = DicomConnector(
                    host=do.dataset.to_dicom_location.host,
                    port=do.dataset.to_dicom_location.port,
                    ae_title=do.dataset.to_dicom_location.ae_title,
                )
                dicom_verify = dicom_connector.verify()

                if not dicom_verify:
                    logger.error(
                        "Unable to connect to Dicom Location: {0} {1} {2}".format(
                            do.dataset.to_dicom_location.host,
                            do.dataset.to_dicom_location.port,
                            do.dataset.to_dicom_location.ae_title,
                        )
                    )
                    continue

                send_result = dicom_connector.send_dcm(do.path)

                if send_result:
                    do.is_sent = True
                    db.session.add(do)
                    db.session.commit()

            else:
                logger.warning(
                    "DICOM Data Object output but not Dicom To location defined in Dataset"
                )

    end = time.time()
    time_taken = end - start
    logger.info("Dataset processing complete, took: " + str(time_taken))
    logger.info("Number of data objects generated: " + str(len(output_data_objects)))

    state_details = {
        "current": len(input_objects),
        "total": len(input_objects),
        "status": "Running Algorithm Complete: {0}".format(algorithm_name),
    }

    task.update_state(state="COMPLETE", meta=state_details)
