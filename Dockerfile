FROM ubuntu:18.04

RUN apt-get update; apt-get install -y python3.6-dev python3-pip redis-server libgl1-mesa-glx libsm6 libxext6 libxrender-dev git

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
RUN python -m pip install --upgrade pip

WORKDIR /home/service

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY requirements-dev.txt requirements-dev.txt
RUN pip install -r requirements-dev.txt

COPY . platipy

COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENV PYTHONPATH "/workspaces"

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV WORK C.UTF-8

ARG dicom_listen_port=7777

ENV DICOM_LISTEN_PORT ${dicom_listen_port}
ENV DICOM_LISTEN_AET PLATIPY_SERVICE

RUN printf '#!/bin/bash\npython3 -m platipy.backend.manage $@\n' > /usr/bin/manage && chmod +x /usr/bin/manage

EXPOSE 8000
EXPOSE ${dicom_listen_port}

ENV WORK /data
RUN mkdir /logs /data && chmod 0777 /logs /data

ENTRYPOINT ["/entrypoint.sh"]

CMD [ "manage" ]
