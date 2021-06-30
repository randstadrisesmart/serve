from abc import ABC
import json
import logging
import os
import ast
import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering,AutoModelForTokenClassification

from ts.torch_handler.base_handler import BaseHandler

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
        self.nlp = spacy.load('en_core_web_sm')
        model_pt_path = os.path.join(model_dir, serialized_file)
        #self.device = torch.device("cuda:0")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
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

        if not os.path.isfile(os.path.join(model_dir, "vocab.txt")):
            self.tokenizer = AutoTokenizer.from_pretrained(self.setup_config["model_name"],do_lower_case=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir,do_lower_case=False)

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
        #text = str(text).replace("&nbsp", "")
        input_texts = getSentences(self, text.lower().decode('utf-8').replace("&nbsp", ""))
        max_length = self.setup_config["max_length"]
        #logger.info("Received text: '%s'", text)
        inputs_array = []

        
        input_len = 0
        sentences_value = ''
        process_sentences = []
        for sentence in input_texts:
            input_len = input_len + len(sentence.split())
            if input_len < int(max_length)-100:
                sentences_value = sentences_value + ' ' + sentence
            else:
                process_sentences.append(sentences_value)
                sentences_value = sentence
                input_len = len(sentence.split())

        process_sentences.append(sentences_value)
        logger.info("sentences '%s' ", process_sentences)

        for input_text in process_sentences:
            inputs = self.tokenizer.encode_plus(input_text,max_length = int(max_length), add_special_tokens = True, return_tensors = 'pt')
            inputs_array.append(inputs)
        #preprocessing text for question_answering.
        return inputs_array

    def inference(self, inputs_array):
        """ Predict the class (or classes) of the received text using the serialized transformers checkpoint.
        """
        prediction_array = []
        for inputs in inputs_array:
            input_ids = torch.tensor(inputs["input_ids"]).to(self.device)
        # Handling inference for sequence_classification.
            if self.setup_config["mode"]== "token_classification":
                outputs = self.model(input_ids)[0]
                predictions = torch.argmax(outputs, dim=2)
                tokens = self.tokenizer.tokenize(self.tokenizer.decode(inputs["input_ids"][0]))
                if self.mapping:
                    label_list = self.mapping["label_list"]
                label_list = label_list.strip('][').split(', ')
                prediction = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())]
                
                prediction_array.extend(get_skill_list(prediction))
                #skill_value_str = ''
                #last_val = ''
                #for value in prediction:
                    #if 'SKILL' in value[1]:
                        #if  "[SEP]" not in value[0]:
                            #skill_value_str = skill_value_str + ' ' + value[0]
                            #prediction_array.append(value[0])
                    #elif skill_value_str and '##' in value[0]:
                        #skill_value_str = skill_value_str + ' ' + value[0]
                    #elif last_val != '' and 'SKILL' not in last_val[1] and 'I-SKILL' in value[1]:
                        #skill_value_str = last_val[0] + ' ' + value[0]    
                    #else:
                        #if skill_value_str:    
                            #prediction_array.append(skill_value_str.replace(' ##','').strip())
                            #skill_value_str = ''
                    #last_val = value        
                    #else:
                        #if skill_value_str and '##' in value[0]:
                            #skill_value_str = skill_value_str + ' ' + value[0]
                        #else:
                            #if skill_value_str:    
                                #prediction_array.append(skill_value_str.replace(' ##','').strip())
                                #skill_value_str = ''    
                                        
                logger.info("Model predicted: '%s'", prediction)    

        final_skill_array = []
        for skill in prediction_array:
            if skill not in final_skill_array:
                final_skill_array.append(skill)
        logger.info("length of Model predicted: '%s'", prediction_array)
        return [final_skill_array]

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output


_service = TransformersSeqClassifierHandler()

def getSentences(self, description):
    sentences = []
    text_sentences = self.nlp(description)
    
    for sentence in text_sentences.sents:
        #print(sentence.text)
        #logger.info("sentence text: '%s'", sentence.text)
        sentences.append(sentence.text)
    
    return sentences

def get_skill_list(predictions):
    old_val = ''
    complete_word = ''
    complete_skill_word = ''
    skill_list = []
    for prediction in predictions:
        if '##' in prediction[0]:
            if complete_word :
                complete_word = complete_word + prediction[0].replace('##','')
            else:
                complete_word = old_val[0] + prediction[0].replace('##','')
        else:
            #if complete_word :
                #print('')#complete_word.replace('##',''))
            complete_word = ''
        if 'SKILL' in prediction[1]:
            if 'B-SKILL' in prediction[1]:
                if complete_word:
                    complete_skill_word = complete_word
                else:
                    complete_skill_word = complete_skill_word + ' ' + prediction[0]
                #print(complete_skill_word)
            elif 'I-SKILL' in prediction[1] and 'SKILL' in old_val[1]:
                if complete_word:
                    #complete_skill_word = complete_skill_word + ' ' + complete_word
                    if complete_word.count(complete_skill_word.split()[-1]):
                        complete_skill_word = ' '.join(complete_skill_word.split(' ')[:-1]) + ' ' + complete_word
                    else:
                        complete_skill_word = complete_skill_word + ' ' + complete_word
                else:
                    complete_skill_word = complete_skill_word + ' ' + prediction[0]
            elif 'I-SKILL' in prediction[1] and 'SKILL' not in old_val[1]:
                if complete_word:
                    complete_skill_word = old_val[0] + ' ' + complete_word
                else:
                    complete_skill_word = old_val[0] + ' ' + prediction[0]
        else:
            if complete_skill_word:
                skill_list.append(complete_skill_word.strip())
            complete_skill_word = ''    
        old_val=prediction
    return skill_list

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
