FROM platipy/platipy:service

RUN env -C /platipy /root/.local/bin/poetry install -E backend -E nnunet

COPY ./nnUNetTrainerHeart.py /usr/lib/python3.8/site-packages/nnunet/training/network_training/nnUNetTrainerHeart.py

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
