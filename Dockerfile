FROM ubuntu:20.04

USER root

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

RUN apt-get update && apt-get upgrade && \
    apt-get install -y unzip wget curl \
    bash python3.8 python3.8-venv \ 
    python3.9 python3.9-venv python3 python3-venv python3-pip \
    ffmpeg libsm6 libxext6 cmake git protobuf-compiler


# install python v3.7    
RUN echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu focal main" >> /etc/apt/sources.list
RUN apt-key adv --keyserver keyserver.ubuntu.com/ --recv-keys BA6932366A755776
RUN apt update
RUN apt install -y python3.7 python3.7-venv

WORKDIR /workspace 
ENTRYPOINT ["bash"]
