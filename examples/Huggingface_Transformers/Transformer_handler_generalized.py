from abc import ABC
import json
import logging
import os
import ast
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering,AutoModelForTokenClassification
from ts.torch_handler.base_handler import BaseHandler
from transformers import BertTokenizer, BertConfig

logger = logging.getLogger(__name__)

class TransformersSeqClassifierHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """
    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")))
        logger.info("device : ", self.device) 
        #read configs for the mode, model_name, etc. from setup_config.json
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning('Missing the setup_config.json file.')

        #Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        #further setup config can be added.
        if self.setup_config["save_mode"] == "torchscript":
            self.model = torch.jit.load(model_pt_path)
        elif self.setup_config["save_mode"] == "pretrained":
            if self.setup_config["mode"]== "sequence_classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            elif self.setup_config["mode"]== "question_answering":
                self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
            elif self.setup_config["mode"]== "token_classification":
                self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
            else:
                logger.warning('Missing the operation mode.')
        else:
            logger.warning('Missing the checkpoint or state_dict.')

        if not os.path.isfile(os.path.join(model_dir, "vocab.*")):
            self.tokenizer = AutoTokenizer.from_pretrained(self.setup_config["model_name"],do_lower_case=self.setup_config["do_lower_case"])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir,do_lower_case=self.setup_config["do_lower_case"])

        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        self.model.to(self.device)
        self.model.eval()

        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))

        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        # Question answering does not need the index_to_name.json file.
        if not self.setup_config["mode"]== "question_answering":
            if os.path.isfile(mapping_file_path):
                with open(mapping_file_path) as f:
                    self.mapping = json.load(f)
            else:
                logger.warning('Missing the index_to_name.json file.')

        self.initialized = True

    def preprocess(self, data):
        """ Basic text preprocessing, based on the user's chocie of application mode.
        """
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        input_text = text.decode('utf-8')
        max_length = self.setup_config["max_length"]
        logger.info("Received text: '%s'", input_text)
        #preprocessing text for sequence_classification and token_classification.
        if self.setup_config["mode"]== "sequence_classification" or self.setup_config["mode"]== "token_classification" :
            inputs = self.tokenizer.encode_plus(input_text,max_length = int(max_length),pad_to_max_length = True, add_special_tokens = True, return_tensors = 'pt')

        return inputs

    def inference(self, inputs):
        """ Predict the class (or classes) of the received text using the serialized transformers checkpoint.
        """


        input_ids = inputs["input_ids"].to(self.device) #inputs["input_ids"].to(torch.device('cuda:0')) #inputs["input_ids"].to(self.device)
        #logger.info("device : ", self.device)
        
        # Handling inference for token_classification.
        if self.setup_config["mode"]== "token_classification":
            with torch.no_grad():
                outputs = self.model(input_ids)[0]
            #predictions = np.argmax(outputs.to('cpu').numpy(), axis=2)
            predictions = torch.argmax(outputs, dim=2)
            #logger.info("output : '%s'",predictions)
            tokens = self.tokenizer.tokenize(self.tokenizer.decode(inputs["input_ids"][0]))
            #tokens = self.tokenizer.tokenize(self.tokenizer.decode(inputs["input_ids"].to('cpu').numpy()[0]))
            #logger.info("tokens : '%s'" , tokens)
            #label_list = ''
            if self.mapping:
                label_list = self.mapping["label_list"]
            label_list = label_list.strip('][').split(', ')
            #logger.info("label list : '%s'",label_list)
            prediction = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())]

            logger.info("Model predicted: '%s'", prediction)

        return [prediction]

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output


_service = TransformersSeqClassifierHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e
