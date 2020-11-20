FROM tensorflow/tensorflow:latest-gpu

MAINTAINER owner

RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && apt-get -y update \
    && add-apt-repository universe

RUN apt-get -y update
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y update
RUN apt-get -y install python3.8
RUN apt-get -y install libgl1-mesa-glx
RUN python3 -m pip install --upgrade pip
RUN pip install numpy
RUN pip install matplotlib
RUN pip install mss
RUN pip install scikit-learn
RUN pip install opencv-python
RUN pip install tqdm
RUN pip install matplotlib
RUN pip install inputs
RUN apt-get -y upgrade

WORKDIR /usr/local/bin/fighter_bot 1.0/  

COPY . .


CMD ["python3", "./main.py"]


# TensorBoard
EXPOSE 6006
