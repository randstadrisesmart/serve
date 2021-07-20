FROM pytorch/torchserve:0.1.1-cuda10.1-cudnn7-runtime

USER 0

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install -y python3-distutils cuda

RUN pip3 install -U pip setuptools wheel
RUN pip3 install cython
RUN pip3 install spacy
RUN pip3 install captum

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN python3 -m spacy download en_core_web_sm

COPY . serve/

RUN python /home/model-server/serve/ts_scripts/install_dependencies.py --cuda=cu102
RUN pip3 install transformers
RUN torch-model-archiver --model-name bert-ner --version 1.0 --serialized-file /home/model-server/serve/final_ner_v2_gpu.pth --handler /home/model-server/serve/examples/Huggingface_Transformers/Transformer_handler_generalized.py --extra-files "/home/model-server/serve/examples/Huggingface_Transformers/setup_config.json,/home/model-server/serve/examples/Huggingface_Transformers/Token_classification_artifacts/index_to_name.json"

RUN mv bert-ner.mar model-store/bert-ner.mar

RUN rm /usr/local/bin/dockerd-entrypoint.sh

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh

RUN ["chown", "root:root", "/usr/local/bin/dockerd-entrypoint.sh"]

RUN ["chmod", "+x", "/usr/local/bin/dockerd-entrypoint.sh"]

USER 1000

#ENTRYPOINT ["ls","-l", "usr/local/bin/dockerd-entrypoint.sh"]

#TRYPOINT ["tail","-f","/dev/null"]
