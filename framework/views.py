from .app import web_app
from loguru import logger

from flask import Flask, request, render_template, send_from_directory


@web_app.route('/endpoint/add', methods=['GET'])
def add_endpoint():

    return render_template('endpoint_add.html', data=web_app.data)

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
            task = listen_task.AsyncResult(endpoint['task_id'])
            status = task.info.get('status', '')
            if 'Error' in status:
                kill_task(endpoint['task_id'])

    return render_template('endpoint_view.html', data=web_app.data, endpoint=endpoint, status=status, format_settings=lambda x: json.dumps(x, indent=4))

@web_app.route('/')
def status():
    """Homepage of the web app, supplying an overview of the status of the system"""
    # celery = Celery('vwadaptor',
    #                broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

    celery_running = False
    if celery.control.inspect().active():
        celery_running = True
    status_context = {'celery': celery_running}
    status_context['algorithms'] = web_app.algorithms

    return render_template('status.html', data=web_app.data, status=status_context)

