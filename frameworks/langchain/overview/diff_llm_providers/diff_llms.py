import os
from types import SimpleNamespace

from langchain.llms import Cohere, __all__
from langchain_community.llms.ctransformers import CTransformers
from langchain_core.prompts import PromptTemplate
# from langchain_cohere.llms import Cohere
from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from common.cfg import *


current_dir = Path(__file__).resolve().parent
current_dir_parent = current_dir.parent


# https://python.langchain.com/v0.2/docs/integrations/providers/

chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo')
# https://platform.openai.com/docs/deprecations
gpt3 = OpenAI(model_name='gpt-3.5-turbo-instruct')
# https://python.langchain.com/v0.2/docs/integrations/providers/cohere/
cohere = Cohere(model='command-xlarge')
# https://python.langchain.com/v0.2/docs/integrations/providers/serpapi/

# https://huggingface.co/bigscience/bloom-1b7
bloom = HuggingFacePipeline.from_model_id(
    model_id="bigscience/bloom-1b7",
    task="text-generation",
    model_kwargs={
        "temperature": 0.1, "max_length": 64,
        "do_sample": True
    }
)

human_msg_txt = 'What is a AI applications developer roadmap in 2024?'
human_msg = HumanMessage(content=human_msg_txt)

def ask_qns_via_model():
    qn_answers = SimpleNamespace()
    try:
        qn_answers.chatgpt_res = chatgpt([human_msg])
        qn_answers.gpt3_res = gpt3(human_msg_txt)
        qn_answers.cohere_res = cohere(human_msg_txt)
        qn_answers.bloom_res = bloom(human_msg_txt)

        # Run with a GPU
        TEMPLATE = """<s>[INST] You are a academic chat bot who's need to help answer the
        user question:
        {user_input} [/INST] </s>
        """
        # https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF
        mistral = CTransformers(
            model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            prompt = PromptTemplate(template=TEMPLATE, input_variables=["user_input"]))

        qn_answers.mistral_res = mistral(human_msg_txt)
    except Exception as ex:
        print(ex)
        # raise ex
    finally:
        dump_data(qn_answers, current_dir / f'{ask_qns_via_model.__name__}.pkl')


def execute():
    ask_qns_via_model()


if __name__ == '__main__':
    execute()
