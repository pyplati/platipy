FROM impit

RUN apt-get update &&\
    apt-get install -y cmake git build-essential && \
    mkdir se && \
    cd se && \
    git clone https://github.com/SuperElastix/SimpleElastix.git && \
    mkdir build && \
    cd build && \
    cmake ../SimpleElastix/SuperBuild && \
    make -j4 && \
    cd SimpleITK-build/Wrapping/Python && \
    pip3 uninstall --yes SimpleITK && \
    python3 Packaging/setup.py install && \
    cd ../../../../.. && \
    rm -rf se
    
COPY . .

ENV ATLAS_PATH /atlas

ENV FLASK_APP service.py
