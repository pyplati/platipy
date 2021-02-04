#!/usr/bin/env bash

# Run redis-server
redis-server --daemonize yes

# Run celery beat and worker
celery beat --app=service:celery --loglevel=INFO &
celery worker --app=service:celery --loglevel=INFO &

# Run the gunicorn server
exec gunicorn -b :8000 --timeout 300 --graceful-timeout 60 --access-logfile - --error-logfile - service:app
