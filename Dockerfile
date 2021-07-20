FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

ARG CUDA_VERSION_TO_INSTALL=cu102
ARG TORCH_VER=1.8.1
ARG TORCH_VISION_VER=0.9.1
ARG TORCH_TEXT_VER=0.9.1
ARG TORCH_SERVE_VER=0.1.1
ARG TORCH_MODEL_VER=0.1.1


RUN apt-get update -y
RUN apt-get install -y python3-pip python3-distutils

RUN pip3 install --no-cache-dir torch==$TORCH_VER torchvision==$TORCH_VISION_VER -f https://download.pytorch.org/whl/torch_stable.html; 

RUN pip3 install --no-cache-dir captum torchtext==$TORCH_TEXT_VER torchserve==$TORCH_SERVE_VER torch-model-archiver==$TORCH_MODEL_VER

COPY . serve/

ENTRYPOINT ["tail","-f","/dev/null"]
