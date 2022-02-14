#!/usr/bin/env bash

# Run redis-server
redis-server --daemonize yes

# Run celery beat and worker
celery --app=service:celery beat --loglevel=INFO &
celery --app=service:celery worker --loglevel=INFO &

# Run the gunicorn server
#exec gunicorn -b :8000  --certfile=service.crt --keyfile=service.key --timeout 300 --graceful-timeout 60 --access-logfile - --error-logfile - service:app
exec gunicorn -b :8000  --timeout 300 --graceful-timeout 60 --access-logfile - --error-logfile - service:app
