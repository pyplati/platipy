from flask import Flask, request, render_template, send_from_directory
from flask_restful import Resource, Api, reqparse
from celery import Celery
from celery import current_app    
from celery.bin import worker
from celery.task.control import revoke
import multiprocessing
from multiprocessing import Process
from impit.dicom.communication import DicomConnector
from loguru import logger
import json
import os
import time
import shutil

logger.add("logfile.log")

class ImagingAlgorithm():

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

    working_dir = "./data"
    file_data = "data.json" 
    data = {}
    algorithms = {}
    celery_started = False

    def __init__(self, name):

        super().__init__(name)

    def save_data(self):

        file_data_path = os.path.join(self.working_dir, self.file_data)

        with open(file_data_path, 'w') as outfile:
            json.dump(self.data, outfile, indent=4)

    def register(self, name, default_settings=None):

        def decorator(f):
            self.algorithms.update(
                {name: ImagingAlgorithm(name, f, default_settings)})
            return f

        return decorator

    def init_app(self):

        # Working directory
        # TODO put this in a config file
        if not os.path.exists(self.working_dir):
            os.mkdir(self.working_dir)

        file_data_path = os.path.join(self.working_dir, self.file_data)
        if not os.path.exists(file_data_path):
            self.data['endpoints'] = []
            self.save_data()

        with open(file_data_path) as json_file:
            self.data = json.load(json_file)

        # Clear endpoint tasks as after restart
        for e in self.data['endpoints']:
            if 'task_id' in e:
                # Revoke it incase it still exists
                revoke(e['task_id'], terminate=True)

                # And remove it from the dict
                e.pop('task_id', None)
        self.save_data()

    def run_celery(self):

        if self.celery_started:
            return

        application = current_app._get_current_object()

        celery_worker = worker.worker(app=application)

        options = {
            'broker': web_app.config['CELERY_BROKER_URL'],
            'loglevel': 'INFO',
            'traceback': True,
        }

        celery_worker.run(**options)

    def run(self, host=None, port=None, debug=None,
            load_dotenv=True, **options):

        self.init_app()

        logger.info('Starting APP!')

        p = Process(target=self.run_celery)
        p.start()
        self.celery_started = True

        super().run(host=host, port=port, debug=debug,
                    load_dotenv=load_dotenv, use_reloader=False, **options)

        p.join()

    def test_client(self, use_cookies=True, **kwargs):

        self.init_app()

        return super().test_client(use_cookies=use_cookies, **kwargs)


web_app = FlaskApp(__name__)
web_app.config['SECRET_KEY'] = 'top-secret!'

# Celery configuration
# TODO Should be in a configuration file
web_app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
web_app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
celery = Celery(__name__, broker='redis://localhost:6379/0')
celery.conf.update(web_app.config)

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

    image_dir = os.path.join(web_app.working_dir, "images")
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

        algorithm = web_app.algorithms[endpoint['endpointAlgorithm']]
        if 'settings' in endpoint:
            dicom_output_path = algorithm.function(
                dicom_path, endpoint['settings'])
        else:
            dicom_output_path = algorithm.function(dicom_path)

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
def listen_task(task, endpoint, task_id=None):
    """
    Background task that listens at a specific port for incoming dicom series
    """

    try:
        dicom_connector = DicomConnector(port=int(endpoint['fromPort']))

        dicom_target = DicomConnector(host=endpoint['toHost'], port=int(
            endpoint['toPort']), ae_title=endpoint['toAETitle'])

        dicom_verify = dicom_target.verify()

        task.update_state(state='PROGRESS',
                        meta={'current': 0, 'total': 1,
                                'status': 'Listening at port: {0}'.format(endpoint['fromPort'])})

        def image_recieved(dicom_path):
            logger.info('Image Series Recieved: {0}'.format(dicom_path))

            algorithm = web_app.algorithms[endpoint['endpointAlgorithm']]
            if 'settings' in endpoint:
                dicom_output_path = algorithm.function(
                    dicom_path, endpoint['settings'])
            else:
                dicom_output_path = algorithm.function(dicom_path)

            send_status = dicom_target.send_dcm(dicom_output_path)
            logger.info('Send got status: {0}'.format(send_status))

            # Remove Dicom files
            logger.info('Removing Dicom Path: {0}'.format(dicom_path))
            shutil.rmtree(dicom_path)

        dicom_connector.listen(image_recieved)
    except Exception as e:
        logger.error('Listener Error: ' + str(e))

        # Stop the listen task
        celery.control.revoke(task_id, terminate=True)

        return {'status': 'Error' + str(e)}

    return {'status': 'Complete'}


# Initialize API
api = Api(web_app)

def kill_task(task_id):
    """Kills the celery task with the given ID, and removes the link to the endpoint if available"""

    logger.info('Killing task: {0}'.format(task_id))

    endpoint = None
    for e in web_app.data['endpoints']:
        if 'task_id' in e and e['task_id'] == task_id:
            endpoint = e

    celery.control.revoke(task_id, terminate=True)

    if endpoint:
        endpoint.pop('task_id', None)
        web_app.save_data()

@web_app.route('/endpoint/add', methods=['GET'])
def add_endpoint():

    return render_template('endpoint_add.html')


@web_app.route('/endpoint/<id>', methods=['GET', 'POST'])
def view_endpoint(id):

    endpoint = None
    for e in web_app.data['endpoints']:
        if e['id'] == int(id):
            endpoint = e

    status = ''
    # Check if the last is still running
    if endpoint['endpointType'] == 'listener':
        if 'task_id' in endpoint:
            task = retrieve_task.AsyncResult(endpoint['task_id'])
            status = task.info.get('status', '')
            if 'Error' in status:
                kill_task(endpoint['task_id'])

    return render_template('endpoint_view.html', data=web_app.data, endpoint=endpoint, status=status, format_settings=lambda x: json.dumps(x, indent=4))

@web_app.route('/')
def status():
    """Homepage of the web app, supplying an overview of the status of the system"""
    #celery = Celery('vwadaptor',
    #                broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

    celery_running = False
    if celery.control.inspect().active():
        celery_running = True
    status_context = {'celery': celery_running}
    status_context['algorithms'] = web_app.algorithms

    return render_template('status.html', data=web_app.data, status=status_context)


# REST Framework Endpoints
class DicomEndpoint(Resource):

    parser = reqparse.RequestParser()
    parser.add_argument('endpointName')
    parser.add_argument('endpointAlgorithm')
    parser.add_argument('endpointType')
    parser.add_argument('settings')
    parser.add_argument('fromHost')
    parser.add_argument('fromPort')
    parser.add_argument('fromAETitle')
    parser.add_argument('toHost')
    parser.add_argument('toPort')
    parser.add_argument('toAETitle')

    def get(self, endpoint_id):
        endpoint = None
        for e in web_app.data['endpoints']:
            if e['id'] == int(endpoint_id):
                endpoint = e

        status = ''
        # Check if the last is still running
        if endpoint['endpointType'] == 'listener':
            if 'task_id' in endpoint:
                task = retrieve_task.AsyncResult(endpoint['task_id'])
                status = task.info.get('status', '')
                if 'Error' in status:
                    kill_task(endpoint['task_id'])

        return {endpoint_id: endpoint}

    def post(self):

        args = self.parser.parse_args()
        
        endpoint = args

        # Settings comes through as JSON so parse to dict
        if type(endpoint['settings']) == str:
            endpoint['settings'] = json.loads(endpoint['settings'])
        endpoint['id'] = len(web_app.data['endpoints'])
        web_app.data['endpoints'].append(endpoint)
        web_app.save_data()

        return {endpoint['id']: endpoint}

class DicomEndpoints(Resource):

    def get(self):
        return web_app.data['endpoints']

class TaskStatus(Resource):

    def get(self, task_id):
        """Get the status of a task given the ID"""
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
        return response


class TriggerEndpoint(Resource):

    parser = reqparse.RequestParser()
    parser.add_argument('seriesUID', action='append')

    def post(self, endpoint_id):
        """Fetch data for a retriever endpoint, or listen for a listener"""

        # Get the endpoint with the given id
        endpoint = None
        for e in web_app.data['endpoints']:
            if e['id'] == int(endpoint_id):
                endpoint = e

        if not endpoint:
            return {'error': 'Endpoint with ID {0} not found'.format(endpoint_id)}, 400

        endpointType = endpoint['endpointType']
        if endpointType == 'retriever':

            args = self.parser.parse_args()
            seriesUIDs = args['seriesUID']

            if not seriesUIDs:
                return {'error': 'Supply one or more seriesUID'}, 400

            # Being the retrieving task for this endpoint
            task = retrieve_task.apply_async([endpoint, seriesUIDs])

            # Return JSON data detailing where to poll for updates on the task
            return {'poll': api.url_for(TaskStatus, task_id=task.id), 'type': endpointType}
        else:

            if 'task_id' in endpoint:
                # If a task ID exists, the endpoint is running so stop it
                kill_task(endpoint['task_id'])
            else:
                # If no task ID exists, start a task to begin listening
                logger.debug('Will Listen')
                task = listen_task.apply_async([endpoint])
                logger.debug('Listening')
                endpoint['task_id'] = task.id
                web_app.save_data()

        return {'type': endpointType}


class Algorithm(Resource):

    def get(self):

        result = []
        for a in web_app.algorithms:
            result.append({'name': a, 'settings': web_app.algorithms[a].default_settings})
        return result

class RawData(Resource):

    def get(self):
        """Returns the raw data which has been previously generated"""
        
        return send_from_directory('/impit/framework/imaging/sample/data',
                               'tmp.nii.gz', as_attachment=True)

api.add_resource(DicomEndpoint, '/api/endpoint/<string:endpoint_id>', '/api/endpoint')
api.add_resource(DicomEndpoints, '/api/endpoints')
api.add_resource(TaskStatus, '/api/status/<string:task_id>')
api.add_resource(TriggerEndpoint, '/api/trigger/<string:endpoint_id>')
api.add_resource(Algorithm, '/api/algorithms')
api.add_resource(RawData, '/api/raw')

