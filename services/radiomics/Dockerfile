FROM platipy/platipy:service

COPY requirements-radiomics.txt requirements-radiomics.txt

RUN pip install -r requirements-radiomics.txt

COPY . .

ENV FLASK_APP service.py
