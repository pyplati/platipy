FROM impit

COPY requirements-pinnacle.txt requirements-pinnacle.txt

RUN pip3 install -r requirements-pinnacle.txt

COPY . .

ENV FLASK_APP service.py
