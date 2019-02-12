#!/bin/sh

ps auxww | grep '[c]elery worker' | awk '{print $2}' | xargs kill -9

celery worker -A sample &

python sample.py
