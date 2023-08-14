FROM platipy/platipy:service

COPY requirements-totalseg.txt requirements-totalseg.txt

ENV FLASK_APP service.py

RUN pip install -r requirements-totalseg.txt

COPY . .

ENV TOTALSEG_DISABLE_HTTP1=1

RUN totalseg_download_weights
