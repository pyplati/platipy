FROM platipy/platipy

COPY requirements-nnunet.txt requirements-nnunet.txt

RUN pip3 install -r requirements-nnunet.txt

RUN cd /; git clone https://github.com/MIC-DKFZ/nnUNet.git

COPY ./nnUNetTrainerHeart.py /nnUNet/nnunet/training/network_training/nnUNetTrainerHeart.py

RUN cd /nnUNet; pip3 install -e .

COPY . /home/service

ENV FLASK_APP service.py

RUN mkdir /nnunet
RUN mkdir /nnunet/raw
RUN mkdir /nnunet/preprocessed
RUN mkdir /nnunet/trained_models

ENV nnUNet_raw_data_base /nnunet/raw
ENV nnUNet_preprocessed /nnunet/preprocessed
ENV RESULTS_FOLDER /nnunet/trained_models

CMD [ "nnUNet_install_pretrained_model_from_zip" ]
