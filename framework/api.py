from .app import web_app
from loguru import logger

import flask_restful
from flask_restful import Api, reqparse
from flask import request, send_from_directory
import werkzeug
import uuid
import datetime
import json
import os
import tempfile
import time
from functools import wraps
from ..dicom.communication import DicomConnector
from .models import db, AlchemyEncoder, APIKey, Dataset, DataObject, DicomLocation
from .tasks import run_task, retrieve_task, listen_task


class CustomConfig(object):
    RESTFUL_JSON = {'separators': (', ', ': '),
                    'indent': 2,
                    'cls': AlchemyEncoder}


# Initialize API
api = Api(web_app)
web_app.config.from_object('impit.framework.api.CustomConfig')


def authenticate(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not getattr(func, 'authenticated', True):
            return func(*args, **kwargs)

        if 'API_KEY' in request.headers:
            if APIKey.query.filter_by(key=request.headers['API_KEY']).first():
                return func(*args, **kwargs)

        flask_restful.abort(401)
    return wrapper


class Resource(flask_restful.Resource):
    method_decorators = [authenticate]   # applies to all inherited resources

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
        result = web_app.data['endpoints']

        # Insert the status of the listening endpoints
        for r in result:
            if r['endpointType'] == 'listener':

                if 'task_id' in r:
                    r['status'] = 'LISTENING'
                    r['poll'] = '/api/status/{0}'.format(r['task_id'])
                else:
                    r['status'] = 'STOPPED'

        return web_app.data['endpoints']


class TaskStatus(Resource):

    def get(self, task_id):
        """Get the status of a task given the ID"""
        task = run_task.AsyncResult(task_id)
        if task.state == 'PENDING':
            response = {
                'state': task.state,
                'current': 0,
                'total': 1,
                'status': 'Pending...'
            }
        elif task.state != 'FAILURE':

            if task.info:
                response = {
                    'state': task.state,
                    'current': task.info.get('current', 0),
                    'total': task.info.get('total', 1),
                    'status': task.info.get('status', '')
                }
                if 'result' in task.info:
                    response['result'] = task.info['result']
                if 'series' in task.info:
                    response['series'] = task.info['series']
            else:
                response = {'state': task.state}

        else:
            # something went wrong in the background job
            response = {
                'state': task.state,
                'current': 1,
                'total': 1,
                'status': str(task.info),  # this is the exception raised
            }
        return response


# class TriggerEndpoint(Resource):

#     parser = reqparse.RequestParser()
#     parser.add_argument('seriesUID', action='append')
#     parser.add_argument('peer_host')
#     parser.add_argument('peer_port', type=int)
#     parser.add_argument('peer_ae_title')

#     def post(self, endpoint_id):
#         """Fetch data for a retriever endpoint, or listen for a listener"""

#         # Get the endpoint with the given id
#         endpoint = None
#         for e in web_app.data['endpoints']:
#             if e['id'] == int(endpoint_id):
#                 endpoint = e

#         if not endpoint:
#             return {'error': 'Endpoint with ID {0} not found'.format(endpoint_id)}, 400

#         args = self.parser.parse_args()
#         seriesUIDs = args['seriesUID']

#         endpointType = endpoint['endpointType']
#         if endpointType == 'retriever':

#             if not seriesUIDs:
#                 return {'error': 'Supply one or more seriesUID'}, 400

#             # Being the retrieving task for this endpoint
#             task = retrieve_task.apply_async([endpoint, seriesUIDs])

#             # Return JSON data detailing where to poll for updates on the task
#             return {'poll': api.url_for(TaskStatus, task_id=task.id), 'type': endpointType}
#         else:

#             peer_host = args['peer_host']
#             peer_port = args['peer_port']
#             peer_ae_title = args['peer_ae_title']

#             if seriesUIDs or peer_host or peer_port or peer_ae_title:
#                 # If any of these are specified, then move the series from this location
#                 if not seriesUIDs:
#                     return {'error': 'Missing seriesUID: Supply one or more seriesUID to MOVE'}, 400
#                 if not peer_host:
#                     return {'error': 'Missing peer_host: Supply the host of the Dicom location to MOVE from'}, 400
#                 if not peer_port:
#                     return {'error': 'Missing peer_port: Supply the port of the Dicom location to MOVE from'}, 400
#                 if not peer_ae_title:
#                     return {'error': 'Missing peer_ae_title: Supply the AE title of the Dicom host to MOVE from'}, 400

#                 # Check that the listener is listening
#                 if 'task_id' in endpoint:
#                     task_id = endpoint['task_id']
#                 else:
#                     logger.error('Listener not running')
#                     return {'error': 'Endpoint not listening'}, 400

#                 # Being the move task
#                 task = move_task.apply_async(
#                     [endpoint, seriesUIDs, peer_host, peer_port, peer_ae_title])

#                 # Return JSON data detailing where to poll for updates on the task
#                 return {'poll': api.url_for(TaskStatus, task_id=task_id)}

#             status = 'Stopping'
#             if 'task_id' in endpoint:
#                 # If a task ID exists, the endpoint is running so stop it
#                 kill_task(endpoint['task_id'])
#             else:
#                 # If no task ID exists, start a task to begin listening
#                 logger.debug('Will Listen')
#                 task = listen_task.apply_async([endpoint])
#                 logger.debug('Listening')
#                 endpoint['task_id'] = task.id
#                 web_app.save_data()
#                 status = 'Starting'

#         return {'poll': api.url_for(TaskStatus, task_id=task.id), 'type': endpointType, 'status': status}


class DicomLocationEndpoint(Resource):

    parser = reqparse.RequestParser()
    parser.add_argument('name', required=True,
                        help="Name to identify this Dicom location")
    parser.add_argument('host', required=True,
                        help="Dicom location host name or IP address")
    parser.add_argument('port', type=int, required=True,
                        help="The port of the Dicom location")
    parser.add_argument('ae_title', help="AE Title of the Dicom location")
    parser.add_argument(
        'move_ae_title', help="The AE title with which to trigger MOVE command (If set will use Dicom MOVE operation to retrieve, GET if null)")
    parser.add_argument(
        'move_port', help="The Port to recieve MOVE command (Required when move_ae_title is set)")

    def get(self):

        key = request.headers['API_KEY']

        dl = DicomLocation.query.filter_by(owner_key=key).all()

        return dl

    def post(self):

        args = self.parser.parse_args()

        key = request.headers['API_KEY']
        dl = DicomLocation(owner_key=key,
                           name=args['name'],
                           host=args['host'],
                           port=args['port'],
                           ae_title=args['ae_title'],
                           move_ae_title=args['move_ae_title'],
                           move_port=args['move_port'])

        db.session.add(dl)
        db.session.commit()

        return dl


class DataObjectsEndpoint(Resource):

    def get(self):

        key = request.headers['API_KEY']

        do = DataObject.query.filter(
            DataObject.dataset.has(owner_key=key)).all()

        return do


class DataObjectEndpoint(Resource):

    parser = reqparse.RequestParser()
    parser.add_argument('dataset', required=True,
                        help="Dataset ID to add Data Object to")
    parser.add_argument('type', choices=('DICOM', 'FILE'), required=True,
                        help="DICOM for Dicom objects to be fetched from the Dataset Dicom Location. FILE for file sent with request.")
    parser.add_argument('seriesUID')
    parser.add_argument('meta_data')
    parser.add_argument('file_name')
    parser.add_argument(
        'file_data', type=werkzeug.FileStorage, location='files')
    parser.add_argument(
        'parent', help="Data Object ID to which this data object should be linked")

    def get(self, dataobject_id):

        key = request.headers['API_KEY']

        do = DataObject.query.filter(DataObject.dataset.has(
            owner_key=key), DataObject.id == dataobject_id).first()

        if do:
            return do

        return {'Error': 'Data Object not found'}, 404

    def post(self):

        key = request.headers['API_KEY']

        args = self.parser.parse_args()
        dataset_id = args['dataset']

        # Get the dataset to which this data object should be added
        ds = Dataset.query.filter_by(owner_key=key, id=dataset_id).first()
        if not ds:
            return {'Error': 'Dataset not found'}, 404

        # Get the parent dataset if one was given
        parent = None
        if args['parent']:
            parent = DataObject.query.filter_by(
                dataset_id=ds.id, id=args['parent']).first()

            if not parent:
                return {'Error': 'Parent Data Object not found'}, 404

        meta_data = None
        if args['meta_data']:
            meta_data = json.loads(args['meta_data'])

        # Create the DataObject
        do = DataObject(
            dataset=ds,
            is_input=True,
            type=args['type'],
            series_instance_uid=args['seriesUID'],
            meta_data=meta_data,
            parent=parent)
        db.session.add(do)
        db.session.commit()

        if args['type'] == 'DICOM':

            if not ds.from_dicom_location:
                return {'message': {'from_dicom_location': "Dataset From Dicom Location not set, so unable to add DICOM objects"}}, 400

            if not args['seriesUID']:
                return {'message': {'seriesUID': "SeriesUID is required to be able to fetch DICOM objects"}}, 400

            if ds.from_dicom_location.move_ae_title:

                # Fetch Dicom data using MOVE
                # Check whether or not we are listening for for Dicom MOVE
                listening_connector = DicomConnector(
                    host='127.0.0.1',
                    port=ds.from_dicom_location.move_port,
                    ae_title=ds.from_dicom_location.move_ae_title)

                if not listening_connector.verify():

                    # Not listening on the move_port, so start the listen task
                    listen_task.apply_async([
                        ds.from_dicom_location.move_port,
                        ds.from_dicom_location.move_ae_title
                    ])

                    # Verify Dicom Location is listening
                    timeout_seconds = 20
                    time_waited = 0
                    while not listening_connector.verify():
                        logger.debug(
                            'Not listening for MOVE, sleeping for 1 second and will try again')
                        time.sleep(1)
                        time_waited += 1

                        if time_waited >= timeout_seconds:
                            msg = 'Listener for MOVE timeout on port: {0}'.format(
                                ds.from_dicom_location.move_port)
                            logger.error(msg)
                            return {'message': {'from_dicom_location': msg}}, 400

                    logger.info('Listening for MOVE OK')

                # Then trigger MOVE
                logger.info('Triggering MOVE for series UID: {0}'.format(
                    do.series_instance_uid))
                dicom_connector = DicomConnector(
                    host=ds.from_dicom_location.host,
                    port=ds.from_dicom_location.port,
                    ae_title=ds.from_dicom_location.ae_title)

                dicom_verify = dicom_connector.verify()

                if dicom_verify:
                    dicom_connector.move_series(
                        do.series_instance_uid, move_aet=ds.from_dicom_location.move_ae_title)
                else:
                    msg = 'Unable to connect to Dicom Location: {0} {1} {2}'.format(
                        ds.from_dicom_location.host,
                        ds.from_dicom_location.port,
                        ds.from_dicom_location.ae_title)
                    logger.error(msg)
                    return {'message': {'from_dicom_location': msg}}, 400

            else:
                # Fetch Dicom data using GET
                task = retrieve_task.apply_async([do.id])

        elif args['type'] == 'FILE':

            if not args['file_name']:
                return {'message': {'file': "Provide the file name"}}, 400

            if not args['file_data']:
                return {'message': {'file': "Provide the file data"}}, 400

            # Save the file
            file_path = os.path.join(tempfile.mkdtemp(), args['file_name'])
            args['file_data'].save(file_path)
            do.is_fetched = True
            do.path = file_path

            db.session.add(do)
            db.session.commit()

        return do

    def delete(self, dataobject_id):

        key = request.headers['API_KEY']

        do = DataObject.query.filter(DataObject.dataset.has(
            owner_key=key), DataObject.id == dataobject_id).first()

        if do:
            db.session.delete(do)
            db.session.commit()

        return 200


class DataObjectDownloadEndpoint(Resource):

    parser = reqparse.RequestParser()

    def get(self, dataobject_id):
        """Returns the file for the given dataobject"""

        key = request.headers['API_KEY']

        do = DataObject.query.filter(DataObject.dataset.has(
            owner_key=key), DataObject.id == dataobject_id).first()

        if do:

            if not do.type == 'FILE':
                return {'Error': 'Can only download Data Objects of type FILE'}, 400

            f = do.path

            if not os.path.exists(f):
                return {'Error': 'File could not be found, perhaps it has expired'}, 404

            logger.info('Downloading file: {0}'.format(f))
            return send_from_directory(os.path.dirname(f),
                                       os.path.basename(f), as_attachment=True)

        return {'Error': 'Data Object not found'}, 404


class DatasetsEndpoint(Resource):

    def get(self):

        key = request.headers['API_KEY']

        ds = Dataset.query.filter_by(owner_key=key).all()

        return ds


class DatasetEndpoint(Resource):

    parser = reqparse.RequestParser()
    parser.add_argument('from_dicom_location',
                        help='ID of DicomLocation from which to retrieve DICOM data')
    parser.add_argument('to_dicom_location',
                        help='ID of DicomLocation the send output data to')
    parser.add_argument('timeout', type=int, default=24)

    def get(self, dataset_id):

        key = request.headers['API_KEY']

        ds = Dataset.query.filter_by(owner_key=key, id=dataset_id).first()

        if ds:
            return ds

        return {'Error': 'Dataset not found'}, 404

    def post(self):

        args = self.parser.parse_args()

        key = request.headers['API_KEY']
        expiry = datetime.datetime.now(
        ) + datetime.timedelta(hours=args['timeout'])
        ds = Dataset(owner_key=key, timeout=expiry)

        if args['from_dicom_location']:
            from_dicom_location = DicomLocation.query.filter_by(
                owner_key=key, id=args['from_dicom_location']).first()
            if not from_dicom_location:
                return {'Error': 'From Dicom Location not found'}, 404
            ds.from_dicom_location = from_dicom_location

        if args['to_dicom_location']:
            to_dicom_location = DicomLocation.query.filter_by(
                owner_key=key, id=args['to_dicom_location']).first()
            if not to_dicom_location:
                return {'Error': 'To Dicom Location not found'}, 404
            ds.to_dicom_location = to_dicom_location

        db.session.add(ds)
        db.session.commit()

        return ds


class DatasetReadyEndpoint(Resource):

    parser = reqparse.RequestParser()

    def get(self, dataset_id):

        key = request.headers['API_KEY']

        ds = Dataset.query.filter_by(owner_key=key, id=dataset_id).first()

        if ds:
            if len(ds.input_data_objects) == 0:
                return {'ready': False}, 200

            for d in ds.input_data_objects:
                if not d.is_fetched:
                    return {'ready': False}, 200

            return {'ready': True}, 200

        return {'Error': 'Dataset not found'}, 404


class AlgorithmEndpoint(Resource):

    def get(self):

        result = []
        for a in web_app.algorithms:
            result.append(
                {'name': a, 'default_settings': web_app.algorithms[a].default_settings})
        return result


class TriggerEndpoint(Resource):

    parser = reqparse.RequestParser()
    parser.add_argument('algorithm', required=True,
                        help="The name of the algorithm to trigger")
    parser.add_argument('dataset', required=True,
                        help="The ID of the dataset to pass to the algorithm")
    parser.add_argument(
        'config', help="JSON configuration for algorithm. Default configuration will be used if not set.")

    def post(self):

        args = self.parser.parse_args()

        if not args['algorithm'] in web_app.algorithms:
            return {'Error': 'No algorithm found with name: {0}'.format(args['algorithm'])}, 404

        algorithm = web_app.algorithms[args['algorithm']]

        config = algorithm.default_settings
        if args['config']:
            try:
                config = json.loads(args['config'])
            except json.decoder.JSONDecodeError:
                return {'Error': 'Could not parse JSON config given'}, 500

        key = request.headers['API_KEY']
        ds = Dataset.query.filter_by(owner_key=key, id=args['dataset']).first()
        if not ds:
            return {'Error': 'Dataset not found'}, 404

        # Ensure that all input data objects in the dataset have been fetched
        if len(ds.input_data_objects) == 0:
            return {'Error': 'Dataset does not contain any data objects'}, 400

        for d in ds.input_data_objects:
            if not d.is_fetched:
                return {'Error': 'Dataset contains Data Objects that have not been fetched yet. Wait for them to be fetched or DICOM Send them if necessary.'}, 400

        # Start the algorithm task
        task = run_task.apply_async([algorithm.name, config, ds.id])

        # Return JSON data detailing where to poll for updates on the task
        return {'poll': api.url_for(TaskStatus, task_id=task.id)}


api.add_resource(
    DicomEndpoint, '/api/endpoint/<string:endpoint_id>', '/api/endpoint')
api.add_resource(DicomEndpoints, '/api/endpoints')
api.add_resource(TaskStatus, '/api/status/<string:task_id>')
api.add_resource(TriggerEndpoint, '/api/trigger')

api.add_resource(DatasetsEndpoint, '/api/datasets')
api.add_resource(DatasetEndpoint, '/api/dataset',
                 '/api/dataset/<string:dataset_id>')
api.add_resource(DatasetReadyEndpoint,
                 '/api/dataset/ready/<string:dataset_id>')

api.add_resource(DataObjectsEndpoint, '/api/dataobjects')
api.add_resource(DataObjectEndpoint, '/api/dataobject',
                 '/api/dataobject/<string:dataobject_id>')
api.add_resource(DataObjectDownloadEndpoint,
                 '/api/dataobject/download/<string:dataobject_id>')

api.add_resource(AlgorithmEndpoint, '/api/algorithm')

api.add_resource(DicomLocationEndpoint, '/api/dicomlocation',
                 '/api/dicomlocation/<string:dicom_location_id>')
