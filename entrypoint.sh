#!/usr/bin/env bash
#flask db upgrade
#flask translate compile
#exec gunicorn -b :5000 --access-logfile - --error-logfile - microblog:app

# run redis-server
redis-server --daemonize yes

# Run celery beat and worker
celery --app=service:celery beat --loglevel=INFO &
celery --app=service:celery worker --loglevel=INFO &

# Start the Dicom listen celery task
python3 -c "from platipy.backend.tasks import listen_task; listen_task.apply_async([$DICOM_LISTEN_PORT, '$DICOM_LISTEN_AET'])"

exec gunicorn -b :8000 --timeout 300 --graceful-timeout 60 --access-logfile - --error-logfile - service:app

# PID=$!

# # See: http://veithen.github.io/2014/11/16/sigterm-propagation.html
# wait ${PID}
# wait ${PID}
# EXIT_STATUS=$?

# # wait forever
# while true
# do
#   tail -f /dev/null & wait ${!}
# done