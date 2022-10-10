FROM platipy/platipy

RUN apt-get update; DEBIAN_FRONTEND="noninteractive" apt-get install -y redis-server git libgl1-mesa-glx libsm6 libxext6 libxrender-dev libglib2.0-0 pandoc curl

RUN cd /platipy && poetry install --with dev,docs --all-extras
