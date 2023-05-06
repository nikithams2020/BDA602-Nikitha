FROM python:3.9

COPY . /ml_assigment

WORKDIR /ml_assigment

RUN apt-get update && apt-get install -y gcc wget

RUN wget https://dlm.mariadb.com/2678574/Connectors/c/connector-c-3.3.3/mariadb-connector-c-3.3.3-debian-bullseye-amd64.tar.gz -O - | tar -zxf - --strip-components=1 -C /usr

RUN python -m pip install --upgrade pip

RUN python -m pip install -r requirements.txt

RUN chmod +x base.sh

RUN sh base.sh

#RUN python baseball_features.py
