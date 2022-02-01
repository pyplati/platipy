export REDIS_HOST=redis
export REDIS_PORT=6379

celery --app=service:celery beat --loglevel=INFO &
celery --app=service:celery worker --loglevel=INFO &
