FROM platipy/platipy

RUN apt-get update; DEBIAN_FRONTEND="noninteractive" apt-get install -y redis-server

COPY . /code
RUN env -C /code /root/.local/bin/poetry install -E backend

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

ENTRYPOINT ["/entrypoint.sh"]

CMD [ "manage" ]
