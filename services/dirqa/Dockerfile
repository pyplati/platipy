FROM platipy/platipy:service

RUN apt update; apt install -y plastimatch

COPY requirements-dirqa.txt requirements-dirqa.txt

RUN pip3 install -r requirements-dirqa.txt

COPY . .

ENV FLASK_APP service.py
