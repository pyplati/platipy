FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt update; apt install -y python3 python3-pip

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install platipy[backend]

COPY requirements-totalseg.txt requirements-totalseg.txt

ENV FLASK_APP service.py

RUN pip install -r requirements-totalseg.txt

ENV TOTALSEG_DISABLE_HTTP1=1

RUN totalseg_download_weights

RUN apt install -y redis-server
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ARG dicom_listen_port=7777

ENV DICOM_LISTEN_PORT ${dicom_listen_port}
ENV DICOM_LISTEN_AET PLATIPY_SERVICE

RUN printf '#!/bin/bash\npython3 -m platipy.backend.manage $@\n' > /usr/bin/manage && \
    chmod +x /usr/bin/manage

EXPOSE 8000
EXPOSE ${dicom_listen_port}

ENV PYTHONPATH "/home/service"
WORKDIR /home/service

ENV WORK /data
RUN mkdir /logs /data && chmod 0777 /logs /data

RUN pip install "SQLAlchemy<2.0.0"

ENTRYPOINT ["/entrypoint.sh"]

CMD [ "manage" ]

COPY . .
