FROM pytorch/torchserve

COPY . serve/
RUN ls -la
RUN ls -la serve/
RUN pwd
RUN pip3 install -U pip setuptools wheel
RUN pip3 install cython
RUN pip3 install -U pip setuptools wheel
RUN pip3 install spacy

RUN export LC_ALL=C.UTF-8
RUN export LANG=C.UTF-8
RUN python3 -m spacy download en_core_web_sm

RUN pip3 install transformers
RUN torch-model-archiver --model-name bert-ner --version 1.0 --serialized-file /home/model-server/serve/final_ner_v2_gpu.pth --handler /home/model-server/serve/examples/Huggingface_Transformers/Transformer_handler_generalized.py --extra-files "/home/model-server/serve/examples/Huggingface_Transformers/setup_config.json,/home/model-server/serve/examples/Huggingface_Transformers/Token_classification_artifacts/index_to_name.json"

RUN mv bert-ner.mar model-store/bert-ner.mar

USER 0

RUN rm /usr/local/bin/dockerd-entrypoint.sh

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh

RUN ["chown", "root:root", "/usr/local/bin/dockerd-entrypoint.sh"]

RUN ["chmod", "+x", "/usr/local/bin/dockerd-entrypoint.sh"]

USER 1000

#ENTRYPOINT ["ls","-l", "usr/local/bin/dockerd-entrypoint.sh"]

#TRYPOINT ["tail","-f","/dev/null"]
