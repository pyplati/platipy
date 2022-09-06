FROM ubuntu:20.04

RUN apt-get update; DEBIAN_FRONTEND="noninteractive" apt-get install -y redis-server git python3-pip libgl1-mesa-glx libsm6 libxext6 libxrender-dev libglib2.0-0 pandoc curl

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip

COPY poetry.lock /platipy/poetry.lock
COPY pyproject.toml /platipy/pyproject.toml

RUN curl -sSL https://install.python-poetry.org | python -  --version 1.2.0rc1
RUN echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc 
RUN /root/.local/bin/poetry config virtualenvs.create false
RUN env -C /platipy /root/.local/bin/poetry install --without dev,docs
