FROM platipy/platipy

COPY requirements-radiomics.txt requirements-radiomics.txt

RUN pip3 install -r requirements-radiomics.txt

COPY . .

ENV FLASK_APP service.py
