FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git \
    wget \
    nano \ 
    libglib2.0-0 \
    python3 \
    python3-dev \
    python3-pip \
    freeglut3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install matplotlib tqdm open3d pillow numpy ipykernel opencv-python

RUN git clone --recurse-submodules https://github.com/tauzn-clock/InformationOptimisation
RUN cd /InformationOptimisation/segment-anything && pip3 install -e .
WORKDIR /InformationOptimisation

