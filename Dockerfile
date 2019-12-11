FROM ubuntu:latest

LABEL maintainer="Adam Levin <adamlevin44@gmail.com>"
ENV TZ=

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
RUN apt-get update
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt update
RUN apt-get install -y gcc-9 libomp-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git-all cmake
RUN mkdir project

COPY requirements.txt /project/requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r /project/requirements.txt

RUN apt-get -y install g++-9

RUN cd project && \
git clone --recursive https://github.com/dmlc/xgboost && \
cd xgboost && \
mkdir build && \
cd build && \
CC=gcc-9 CXX=g++-9 cmake .. && \
make -j4 && \
cd /project/xgboost/python-package && \
python3 setup.py install

COPY notebooks /project/notebooks
COPY scripts /project/scripts
COPY tags_train /project/tags_train
COPY tags_test /project/tags_test
COPY features_test /project/features_test
COPY features_train /project/features_train
COPY descriptions_train /project/descriptions_train
COPY descriptions_test /project/descriptions_test

WORKDIR /project

CMD ["jupyter", "notebook", "--allow-root","--ip=0.0.0.0", "--no-browser"]
