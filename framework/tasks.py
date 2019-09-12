from impit.framework import celery, db, app
from celery.schedules import crontab

from loguru import logger
import tempfile
import pydicom
import os
import datetime
import shutil
import time

from ..dicom.communication import DicomConnector

from .models import Dataset, DataObject

celery.conf.beat_schedule = {
    # Executes every hour
    'clean-up-every-hour': {
        'task': 'impit.framework.tasks.clean_up_task',
        'schedule': crontab(minute=0),
        'args': (),
    },
}

celery.conf.timezone = 'UTC'

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
                            logger.debug('Removing Directory: ', do.path)
                            shutil.rmtree(do.path)
                        elif os.path.isfile(do.path):
                            logger.debug('Removing File: ', do.path)
                            os.remove(do.path)
                        else:
                            logger.debug('Data missing, must have already been deleted: ', do.path)

                        num_data_objs_removed += 1

                        do.is_fetched = False
                        do.path = None
                        db.session.commit()

                except Exception as e:
                    logger.warning("Exception occured when removing DataObject: " + str(do))

    logger.info("Clean Up Task Complete: Removed {0} DataObjects", num_data_objs_removed)

@celery.task(bind=True)
def retrieve_task(task, data_object_id):
    """
    Fetch a Dicom Object from the dicom location using the retrieve_type GET or MOVE
    """

    do = DataObject.query.filter_by(id=data_object_id).first()

    dicom_connector = DicomConnector(host=do.dataset.from_dicom_location.host,
                                     port=do.dataset.from_dicom_location.port,
                                     ae_title=do.dataset.from_dicom_location.ae_title)
    dicom_verify = dicom_connector.verify()

    if not dicom_verify:
        logger.error('Unable to connect to Dicom Location: {0} {1} {2}'.format(
            do.dataset.from_dicom_location.host, do.dataset.from_dicom_location.port, do.dataset.from_dicom_location.ae_title))
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

    task.update_state(state='PROGRESS',
                      meta={'current': count, 'total': total,
                            'status': 'Verifying dicom location'})

    dicom_verify = dicom_connector.verify()

    if dicom_verify == None:
        return {'current': 100, 'total': 100, 'status': 'Unable to connect to dicom location'}

    for suid in seriesUIDs:

        task.update_state(state='PROGRESS',
                          meta={'current': count, 'total': total,
                                'status': 'Moving series for UID: {0}'.format(suid)})

        logger.info('Moving Series with UID: {0}'.format(suid))
        dicom_connector.move_series(suid)

        count = count + 1

    task.update_state(state='SUCCESS',
                      meta={'current': total, 'total': total,
                            'status': 'Move Complete'})


@celery.task(bind=True)
def listen_task(task, listen_port, listen_ae_title):
    """
    Background task that listens at a specific port for incoming dicom series
    """
    
    logger.info('Starting Dicom Listener on port: {0} with AE Title: {1}',
        listen_port,
        listen_ae_title)

    task_id = task.request.id

    try:
        dicom_connector = DicomConnector(
            port=listen_port, ae_title=listen_ae_title)

        def series_recieved(dicom_path):
            logger.info(
                'Series Recieved at path: {0}'.format(dicom_path))

            # Get the SeriesUID
            series_uid = None
            for f in os.listdir(dicom_path):
                f = os.path.join(dicom_path, f)

                try:
                    d = pydicom.read_file(f)
                    series_uid = d.SeriesInstanceUID
                except Exception as e:
                    logger.debug('No Series UID in: {0}'.format(f))
                    logger.debug(e)

            if series_uid:
                logger.info('Image Series UID: {0}'.format(series_uid))
            else:
                logger.error('Series UID could not be determined... Stopping')
                return

            # Find the data objects with the given series UID and update them
            dos = DataObject.query.filter_by(
                series_instance_uid=series_uid).all()

            if len(dos) == 0:
                logger.error(
                    'No Data Object found with Series UID: {0} ... Stopping'.format(series_uid))
                return

            for do in dos:

                do.is_fetched = True
                do.path = dicom_path
                db.session.commit()

        if not listen_ae_title:
            listen_ae_title = 'PYNETDICOM'

        dicom_connector.listen(series_recieved, ae_title=listen_ae_title)

    except Exception as e:
        logger.error('Listener Error: ' + str(e))

        # Stop the listen task
        celery.control.revoke(task_id, terminate=True)

    return {'status': 'Complete'}


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

    logger.info('Will run algorithm: ' + algorithm_name)
    logger.info('Using settings: ' + str(config))
    logger.info('Number of data objects in dataset: ' + str(len(input_objects)))

    state_details = {'current': 0, 'total': len(input_objects),
                     'status': 'Running Algorithm: {0}'.format(algorithm_name)}

    task.update_state(state='RUNNING', meta=state_details)

    if config is None:
        output_data_objects = algorithm.function(
            input_objects, tempfile.mkdtemp())
    else:
        output_data_objects = algorithm.function(
            input_objects, tempfile.mkdtemp(), config)

    if not output_data_objects:
        logger.warning(
            'Algorithm ({0}) did not return any output objects'.format(algorithm_name))

    # Save the data objects
    for do in output_data_objects:
        do.dataset_id = ds.id
        db.session.add(do)
        db.session.commit()

        if do.type == 'DICOM':
            if ds.to_dicom_location:

                logger.info('Sending to Dicom To Location')
                dicom_connector = DicomConnector(host=do.dataset.to_dicom_location.host,
                                                 port=do.dataset.to_dicom_location.port,
                                                 ae_title=do.dataset.to_dicom_location.ae_title)
                dicom_verify = dicom_connector.verify()

                if not dicom_verify:
                    logger.error('Unable to connect to Dicom Location: {0} {1} {2}'.format(
                        do.dataset.to_dicom_location.host,
                        do.dataset.to_dicom_location.port,
                        do.dataset.to_dicom_location.ae_title))
                    continue

                send_result = dicom_connector.send_dcm(do.path)

                if send_result:
                    do.is_sent = True
                    db.session.add(do)
                    db.session.commit()

            else:
                logger.warning(
                    'DICOM Data Object output but not Dicom To location defined in Dataset')

    end = time.time()
    time_taken = end - start
    logger.info('Dataset processing complete, took: ' + str(time_taken))
    logger.info('Number of data objects generated: ' + str(len(output_data_objects)))

    state_details = {'current': len(input_objects), 'total': len(input_objects),
                     'status': 'Running Algorithm Complete: {0}'.format(algorithm_name)}

    task.update_state(state='COMPLETE', meta=state_details)
