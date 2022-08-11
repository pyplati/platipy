#!/usr/bin/env bash

# Run redis-server
redis-server --daemonize yes

# Run celery beat and worker
celery --app=service:celery beat --loglevel=INFO &
celery --app=service:celery worker --loglevel=INFO &

# Start the DICOM listener for the service
celery --app=service:celery call platipy.backend.tasks.run_dicom_listener

# Run the gunicorn server
CERT_FILE=service.crt
KEY_FILE=service.key
if [ -f "$CERT_FILE" ]; then
    echo "SSL Certificates Found. Will serve over HTTPS."
    exec gunicorn -b :8000  --certfile=service.crt --keyfile=service.key --timeout 300 --graceful-timeout 60 --access-logfile - --error-logfile - service:app
else
    echo "WARNING: No SSL certificates found. Generate them with 'manage ssl'."
    echo "Running without SSL, not suitable for production use."
    exec gunicorn -b :8000  --timeout 300 --graceful-timeout 60 --access-logfile - --error-logfile - service:app
fi
