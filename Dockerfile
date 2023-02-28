FROM ubuntu:20.04

RUN apt-get update; DEBIAN_FRONTEND="noninteractive" apt-get install -y python3-pip libgl1-mesa-glx libsm6 libxext6 libxrender-dev libglib2.0-0 curl

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --upgrade pip

COPY poetry.lock /platipy/poetry.lock
COPY pyproject.toml /platipy/pyproject.toml

RUN curl -sSL https://install.python-poetry.org | python -  --version 1.3.2
RUN echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
RUN echo "/usr/lib/python3.8/site-packages" >> /usr/local/lib/python3.8/dist-packages/site-packages.pth

ENV PATH="/root/.local/bin:${PATH}"

RUN poetry config virtualenvs.create false
