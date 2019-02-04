#!/bin/sh

ps auxww | grep 'celery worker' | awk '{print $2}' | xargs kill -9

celery worker -A sample &

python sample.py
