import torch
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import OpenAI
from transformers import AutoModelForCausalLM

# Langchain store


def init_llm(is_open_ai: bool = False, is_gpt2 = True, is_ms = False, is_google = False):
    # https://huggingface.co/blog/langchain
    # https://api.python.langchain.com/en/latest/llms/langchain_community.llms.huggingface_pipeline.HuggingFacePipeline.html
    # https://huggingface.co/models

    llm = None
    if is_open_ai:
        llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
    elif is_gpt2:
        # llm1 = HuggingFacePipeline.from_model_id(
        #     # model_id="microsoft/Phi-3-mini-4k-instruct",
        #     model_id="gpt2",
        #     task="text-generation",
        #     pipeline_kwargs={
        #         'max_length': 2500,
        #         # "max_new_tokens": 10000,
        #         # "temperature": 0,
        #     },
        #     # huggingfacehub_api_token=API_KEY_HUGGING_FACE,
        #     # token=API_KEY_HUGGING_FACE
        # )

        llm = HuggingFacePipeline.from_model_id(
            model_id="distilbert/distilgpt2",
            task="text-generation",
            pipeline_kwargs={
                'max_length': 1024,
                # 'max_new_tokens': 512,
                "do_sample": True,
                # "temperature": 0,
            },
        )

        # ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}
        # ov_llm = HuggingFacePipeline.from_model_id(
        #     model_id="gpt2",
        #     task="text-generation",
        #     # backend="openvino",
        #     # model_kwargs={"device": "CPU", "ov_config": ov_config},
        #     pipeline_kwargs={"max_new_tokens": 10},
        # )
        #
        # llm = HuggingFacePipeline.from_model_id(
        #     model_id="gpt2",
        #     task="text-generation",
        #     model_kwargs={
        #         # "temperature": 0.7,
        #         "do_sample": True,
        #         "max_position_embeddings": 2048,
        #         "ignore_mismatched_sizes": True
        #     },
        #     pipeline_kwargs={
        #         # Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
        #         # "max_length": 1000,  # GPT2 which is 1024.
        #         "max_new_tokens": 500, # master
        #         # "max_position_embeddings": 2048,
        #     }
        # )
        # # Optimum OpenVINO is an extension for Optimum library which brings Intel OpenVINO backend for Hugging Face Transformers :hugs:.
        # ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}
        #
        # ov_llm = HuggingFacePipeline.from_model_id(
        #     model_id="gpt2",
        #     task="text-generation",
        #     backend="openvino", # Issue with deps depreced pacjage
        #     model_kwargs={"device": "CPU", "ov_config": ov_config},
        #     pipeline_kwargs={"max_new_tokens": 10},
        # )
    elif is_ms:
        llm = HuggingFacePipeline.from_model_id(
            model_id="microsoft/Phi-3-mini-4k-instruct",
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 500,
                "top_k": 50,
                "temperature": 0.7,
                'do_sample': False
            },
        )
    elif is_google:
        llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path="google/gemma-2-9b",
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

    response = llm.invoke("Once upon a time")
    print(response)
    assert len(response) >= 20

    # Option 1: Use this if you want the generation to involve some randomness, controlled by the temperature parameter.
    # do_sample=True,  # Enable sampling
    # temperature=0.1,  # Set the temperature for sampling

    # Option 2: Use this if you prefer deterministic generation without randomness.
    # do_sample=False,  # Ensure sampling is disabled
    return llm