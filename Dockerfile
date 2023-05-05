FROM python:3.9

COPY . /BDA/ML_project

WORKDIR /BDA/ML_project

RUN python3 -m pip install -r requirements.txt

CMD python3 baseball_features.py
