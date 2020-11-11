FROM platipy/platipy:latest

RUN pip3 install matplotlib==3.1.0 jupyterlab pandas itkwidgets seaborn gitpython

ENV HOME=/home/service

RUN mkdir /home/service/.jupyter

RUN apt update

RUN apt install -y nodejs npm cmake

#RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager itkwidgets

#RUN pip3 install --upgrade jupyterlab-git

#RUN jupyter lab build

EXPOSE 22

ENTRYPOINT cd /jpy && jupyter lab --no-browser --ip=0.0.0.0 --port=8080 --allow-root
