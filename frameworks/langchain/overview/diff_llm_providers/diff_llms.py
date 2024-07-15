import os


from langchain_cohere.llms import Cohere
from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from common.cfg import *

# https://python.langchain.com/v0.2/docs/integrations/providers/

chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo')
gpt3 = OpenAI(model_name='text-davinci-003')
# https://python.langchain.com/v0.2/docs/integrations/providers/cohere/
cohere = Cohere(model='command-xlarge')
# https://python.langchain.com/v0.2/docs/integrations/providers/serpapi/

# https://huggingface.co/bigscience/bloom-1b7
bloom = HuggingFacePipeline.from_model_id(
    model_id="bigscience/bloom-1b7",
    task="text-generation",
    model_kwargs={
        "temperature":0, "max_length":64,
        "do_sample": True
    }
)

g = 1

