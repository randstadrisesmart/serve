FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

ARG CUDA_VERSION_TO_INSTALL=cu101
ARG TORCH_VER=1.5.1
ARG TORCH_VISION_VER=0.6.1

RUN apt-get update -y
RUN apt-get install -y python3-pip python3-distutils

RUN pip3 install --no-cache-dir torch==$TORCH_VER+$CUDA_VERSION_TO_INSTALL torchvision==$TORCH_VISION_VER+$CUDA_VERSION_TO_INSTALL -f https://download.pytorch.org/whl/torch_stable.html; 

RUN pip3 install --no-cache-dir captum torchtext torchserve torch-model-archiver

COPY . serve/

ENTRYPOINT ["tail","-f","/dev/null"]
