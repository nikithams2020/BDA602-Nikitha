FROM python:3.9

COPY ./application/ /application/

WORKDIR /application/

RUN apt-get update && apt-get install -y gcc wget \
     mariadb-client 

RUN wget https://dlm.mariadb.com/2678574/Connectors/c/connector-c-3.3.3/mariadb-connector-c-3.3.3-debian-bullseye-amd64.tar.gz -O - | tar -zxf - --strip-components=1 -C /usr

RUN python -m pip install --upgrade pip

RUN python -m pip install -r requirements.txt

RUN chmod +x base.sh

ENTRYPOINT [ "./base.sh" ] 
