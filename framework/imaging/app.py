from flask import Flask, request, render_template, session, flash, redirect, \
    url_for, jsonify
from celery import Celery
from celery.bin import worker
from celery.task.control import revoke
import multiprocessing
from impit.dicom.communication import DicomConnector
from loguru import logger
import json
import os
import time
import shutil

logger.add("logfile.log")

web_app = Flask(__name__)
web_app.config['SECRET_KEY'] = 'top-secret!'

# Celery configuration
web_app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
web_app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

# Initialize Celery
celery = Celery(web_app.name, broker=web_app.config['CELERY_BROKER_URL'])
celery.conf.update(web_app.config)
    
# Working directory
# TODO put this in a config file
working_dir = "./data"
web_app.working_dir = working_dir
if not os.path.exists(working_dir):
    os.mkdir(working_dir)

file_data = "data.json"
file_data_path = os.path.join(working_dir, file_data)
data = {}

def save_data(data):

    with open(file_data_path, 'w') as outfile:
        json.dump(data, outfile)


if not os.path.exists(file_data_path):
    data['endpoints'] = []
    save_data(data)

with open(file_data_path) as json_file:
    data = json.load(json_file)

    # Clear endpoint tasks as after restart
    for e in data['endpoints']:
        if 'task_id' in e:
            # Revoke it incase it still exists
            revoke(e['task_id'], terminate=True)

            # And remove it from the dict
            e.pop('task_id', None)
    save_data(data)

algorithms = {}


def register(name):

    def decorator(f):
        algorithms.update({name: f})
        return f

    return decorator

@celery.task(bind=True)
def retrieve_task(task, endpoint, seriesUIDs):
    """
    Background task that fetches from the configured location and runs the
    imaging algorithm
    """

    # For each series UID supplied, fetch the image series and run the algorithm
    total = len(seriesUIDs)
    count = 0

    dicom_connector = DicomConnector(host=endpoint['fromHost'], port=int(
        endpoint['fromPort']), ae_title=endpoint['fromAETitle'])

    task.update_state(state='PROGRESS',
                      meta={'current': count, 'total': total,
                            'status': 'Verifying dicom (from) location'})

    dicom_verify = dicom_connector.verify()

    if dicom_verify == None:
        return {'current': 100, 'total': 100, 'status': 'Unable to connect to dicom (from) location'}

    dicom_target = DicomConnector(host=endpoint['toHost'], port=int(
        endpoint['toPort']), ae_title=endpoint['toAETitle'])

    task.update_state(state='PROGRESS',
                      meta={'current': count, 'total': total,
                            'status': 'Verifying dicom (to) location'})

    dicom_verify = dicom_target.verify()

    if dicom_verify == None:
        return {'current': 100, 'total': 100, 'status': 'Unable to connect to dicom (to) location'}

    image_dir = os.path.join(working_dir, "images")
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    for suid in seriesUIDs:

        task.update_state(state='PROGRESS',
                          meta={'current': count, 'total': total,
                                'status': 'Fetching series for UID: {0}'.format(suid)})

        dicom_path = dicom_connector.download_series(suid, None)

        task.update_state(state='PROGRESS',
                        meta={'current': count, 'total': total,
                                'status': 'Running algorithm on image series: {0}'.format(endpoint['endpointAlgorithm'])})

        dicom_output_path = algorithms[endpoint['endpointAlgorithm']](dicom_path)

        task.update_state(state='PROGRESS',
                        meta={'current': count, 'total': total,
                                'status': 'Sending result to target location'})

        send_status = dicom_target.send_dcm(dicom_output_path)
        logger.info('Send got status: {0}'.format(send_status))

        # Remove Dicom files
        logger.info('Removing Dicom Path: {0}'.format(dicom_path))
        shutil.rmtree(dicom_path)

        count += 1
        
    return {'current': total, 'total': total, 'status': 'Complete'}


@celery.task(bind=True)
def listen_task(task, endpoint):
    """
    Background task that listens at a specific port for incoming dicom series
    """

    dicom_connector = DicomConnector(port=int(endpoint['fromPort']))

    dicom_target = DicomConnector(host=endpoint['toHost'], port=int(
        endpoint['toPort']), ae_title=endpoint['toAETitle'])

    dicom_verify = dicom_target.verify()

    task.update_state(state='PROGRESS',
                        meta={'current': 0, 'total': 1,
                            'status': 'Listening at port: {0}'.format(endpoint['fromPort'])})

    def image_recieved(dicom_path):
        logger.info('Image Series Recieved: {0}'.format(dicom_path))

        dicom_output_path = algorithms[endpoint['endpointAlgorithm']](dicom_path)

        send_status = dicom_target.send_dcm(dicom_output_path)
        logger.info('Send got status: {0}'.format(send_status))

        # Remove Dicom files
        logger.info('Removing Dicom Path: {0}'.format(dicom_path))
        shutil.rmtree(dicom_path)

    dicom_connector.listen(image_recieved)
    
    return {'status': 'Complete'}

@web_app.route('/endpoint/add', methods=['GET', 'POST'])
def add_endpoint():

    if request.method == 'POST':

        endpoint = request.form.to_dict()
        endpoint['id'] = len(data['endpoints'])
        data['endpoints'].append(endpoint)
        save_data(data)

    return render_template('endpoint_add.html', data=data, algorithms=algorithms)


@web_app.route('/endpoint/<id>', methods=['GET', 'POST'])
def view_endpoint(id):

    endpoint = None
    for e in data['endpoints']:
        if e['id'] == int(id):
            endpoint = e

    return render_template('endpoint_view.html', data=data, endpoint=endpoint)


@web_app.route('/endpoint/trigger/<id>', methods=['GET', 'POST'])
def tigger_endpoint(id):

    endpoint = None
    for e in data['endpoints']:
        if e['id'] == int(id):
            endpoint = e

    endpointType = endpoint['endpointType']
    if endpointType == 'retriever':

        request_data = json.loads(request.data)
        seriesUIDs = request_data['seriesUIDs'].splitlines()

        if len(seriesUIDs) == 0:
            return jsonify({'error': 'Supply Series UIDs'}), 400

        task = retrieve_task.apply_async([endpoint, seriesUIDs])

        return jsonify({'location': url_for('taskstatus', task_id=task.id), 'type': endpointType}), \
            202, {'location': url_for('taskstatus', task_id=task.id), 'type': endpointType}
    else:

        if 'task_id' in endpoint:
            logger.info('Killing task: {0}'.format(endpoint['task_id']))
            celery.control.revoke(endpoint['task_id'], terminate=True)
            endpoint.pop('task_id', None)
            logger.info(endpoint)
            save_data(data)
        else:
            task = listen_task.apply_async([endpoint])

            endpoint['task_id'] = task.id
            save_data(data)

    return jsonify({'type': endpointType}), 202, {'type': endpointType}


@web_app.route('/status/<task_id>')
def taskstatus(task_id):
    task = retrieve_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 1,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 1),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'current': 1,
            'total': 1,
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)


@web_app.route('/')
def status():
    celery = Celery('vwadaptor',
                    broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

    celery_running = False
    if celery.control.inspect().active():
        celery_running = True
    status_context = {'celery': celery_running}
    status_context['algorithms'] = algorithms

    return render_template('status.html', data=data, status=status_context)

    