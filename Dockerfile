FROM ubuntu:18.04

RUN apt-get update; apt-get install -y python3.6-dev python3-pip redis-server libgl1-mesa-glx libsm6 libxext6 libxrender-dev git

WORKDIR /home/service

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY requirements-dev.txt requirements-dev.txt
RUN pip3 install -r requirements-dev.txt

COPY . impit

COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

RUN export PYTHONPATH="/workspaces"

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV WORK C.UTF-8

ARG dicom_listen_port=7777

ENV DICOM_LISTEN_PORT ${dicom_listen_port}
ENV DICOM_LISTEN_AET IMPIT_SERVICE

RUN printf '#!/bin/bash\npython3 -m impit.framework.manage $@\n' > /usr/bin/manage && chmod +x /usr/bin/manage

EXPOSE 8000
EXPOSE ${dicom_listen_port}

ENV WORK /data
RUN mkdir /logs /data && chmod 0777 /logs /data

ENTRYPOINT ["/entrypoint.sh"]

CMD [ "manage" ]
