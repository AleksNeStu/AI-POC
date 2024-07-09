import torch
from langchain_core.language_models import BaseLanguageModel
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import OpenAI
from transformers import AutoModelForCausalLM
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# https://huggingface.co/blog/langchain
# https://api.python.langchain.com/en/latest/llms/langchain_community.llms.huggingface_pipeline.HuggingFacePipeline.html
# https://huggingface.co/models
# https://huggingface.co/docs/transformers/v4.17.0/en/quicktour

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Custom wrapper for the Hugging Face pipeline
DISTIL_GPT2 = 'distilgpt2'



def split_to_chunks():
    # ValueError: Input length of input_ids is 1024, but `max_length` is set to 1024. This can lead to unexpected behavior. You should consider increasing `max_length` or, better yet, setting `max_new_tokens`.
    pass


def calc_tokens(model_name, prompt):
    # Load tokenizer for the specified model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Tokenize the prompt and count tokens
    tokens_enc = tokenizer.encode(prompt, add_special_tokens=True)

    tokens_ids = tokenizer(prompt, return_tensors='pt', truncation=False).input_ids[0]

    input_ids = tokenizer.encode(prompt, add_special_tokens=True, truncation=True, max_length=1024)

    return len(input_ids)

def get_pipeline(model_name: str = DISTIL_GPT2):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline(model=model, tokenizer=tokenizer)
    return pipe


def get_llm_openai(**kwargs):
    kwargs.update(dict(
        model="gpt-3.5-turbo-instruct",
        temperature=0
    ))
    llm = OpenAI(**kwargs)
    return llm

def get_llm_distilgpt2(**kwargs):
    # temperature=0.1 involve some randomness, controlled by the temperature parameter
    kwargs.update(dict(
        model_id=DISTIL_GPT2,
        task="text-generation",
        pipeline_kwargs={
            "max_length": 1024,
            'max_new_tokens': 50,
            # "do_sample": True,
            # "truncation": True,
            # "temperature": 0,
        },
    ))
    llm = HuggingFacePipeline.from_model_id(**kwargs)
    return llm

def get_llm_gpt2():
    pass


def get_llm_gpt2_ov(**kwargs):
    # Optimum OpenVINO is an extension for Optimum library which brings Intel OpenVINO backend for Hugging Face Transformers :hugs:.
    kwargs.update(dict(
        model_id="gpt2",
        task="text-generation",
        # backend="openvino",
        # model_kwargs={"device": "CPU", "ov_config": ov_config},
        pipeline_kwargs={"max_new_tokens": 10},
    ))
    # Deprecated, deps conflicts to run
    ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}
    llm = HuggingFacePipeline.from_model_id(**kwargs)
    return llm

def get_llm_ms(**kwargs):
    kwargs.update(dict(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 500,
            "top_k": 50,
            "temperature": 0.7,
            "do_sample": False,
        },
    ))
    llm = HuggingFacePipeline.from_model_id(**kwargs)
    return llm


qn1 = "what did"
qn1 = "what did they say about regression in the third lecture?"
ct = calc_tokens(DISTIL_GPT2, qn1)
f = 1
#
#
# llm = get_llm_distilgpt2()
# response = llm.invoke("Once upon a time")
# print(response)
# assert len(response) >= 20
