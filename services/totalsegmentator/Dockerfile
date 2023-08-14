FROM platipy/platipy:service

RUN env -C /platipy /root/.local/bin/poetry install -E backend

COPY . /home/service

ENV FLASK_APP service.py

RUN pip install -r requirements-totalseg.txt

COPY . .

RUN download_pretrained_weights

