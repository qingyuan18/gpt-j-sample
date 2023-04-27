from djl_python import Input, Output
import deepspeed
import torch
import logging
import math
import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer



def load_model(properties):
    tensor_parallel = properties["tensor_parallel_degree"]
    model_location = properties['model_dir']
    if "model_id" in properties:
        model_location = properties['model_id']
    logging.info(f"Loading model in {model_location}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_location)
    
    model = AutoModelForCausalLM.from_pretrained(model_location,
                                                 low_cpu_mem_usage=True)
    if "dtype" in properties:
        if properties["dtype"] == "float16":
            model.to(torch.float16)
        if properties["dtype"] == "bfloat16":
            model.to(torch.bfloat16)
    
    logging.info(f"Starting DeepSpeed init with TP={tensor_parallel}")
    model = deepspeed.init_inference(model,
                                     mp_size=tensor_parallel,
                                     dtype=model.dtype,
                                     replace_method='auto',
                                     replace_with_kernel_inject=True)
    return model.module, tokenizer


model = None
tokenizer = None
generator = None


def run_inference(model, tokenizer, data, params):
    generate_kwargs = params
    tokenizer.pad_token = tokenizer.eos_token
    input_tokens = tokenizer.batch_encode_plus(data,
                                               return_tensors="pt",
                                               padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())
    outputs = model.generate(**input_tokens, **generate_kwargs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def handle(inputs: Input):
    global model, tokenizer
    if not model:
        model, tokenizer = load_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None
    data = inputs.get_as_json()
    
    input_sentences = data["inputs"]
    params = data["parameters"]
    
    outputs = run_inference(model, tokenizer, input_sentences, params)
    result = {"outputs": outputs}
    return Output().add_as_json(result)
