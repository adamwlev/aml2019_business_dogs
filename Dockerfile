FROM ubuntu:latest

LABEL maintainer="Adam Levin <adamlevin44@gmail.com>"

RUN apt update
RUN apt-get update
RUN apt-get install -y libatlas-base-dev
RUN apt-get update
RUN apt-get install -y gfortran
RUN apt-get update
RUN apt-get install -y gcc
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-get update
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get install -y python3.7 python3-pip
RUN mkdir project

COPY requirements.txt /project/requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r /project/requirements.txt

COPY descriptions_test /project/descriptions_test
COPY descriptions_train /project/descriptions_train
COPY features_test /project/features_test
COPY features_train /project/features_train
COPY images_test /project/images_test
COPY images_train /project/images_train
COPY notebooks /project/notebooks
COPY tags_test /project/tags_test
COPY tags_train /project/tags_train

WORKDIR /project

CMD ["jupyter", "notebook", "--allow-root","--ip=0.0.0.0", "--no-browser"]
