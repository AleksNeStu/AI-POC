from types import SimpleNamespace


from langchain.indexes import VectorstoreIndexCreator
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ctransformers import CTransformers
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.prompts import PromptTemplate
# from langchain.llms import Cohere
from langchain_cohere.llms import Cohere
# from langchain_cohere import ChatCohere
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.schema import (
    HumanMessage
)
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer


from common.cfg import *

current_dir = Path(__file__).resolve().parent
current_dir_parent = current_dir.parent

# https://python.langchain.com/v0.2/docs/integrations/providers/

chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1)
# https://platform.openai.com/docs/deprecations
gpt3 = OpenAI(model_name='gpt-3.5-turbo-instruct')
embeddings = OpenAIEmbeddings()
# embedding2 = SentenceTransformer('all-MiniLM-L6-v2')
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

human_msg2_txt = 'Make short summary for this document'

@get_data(load=True)
def ask_questions_using_diff_provider_models():
    # get_func_data_path(ask_questions_using_diff_provider_models)
    # d_path = current_dir / f'{ask_questions_using_diff_provider_models.__name__}.pkl'
    # if load:
    #     return load_data(d_path)
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
        return qn_answers
        # raise ex
    # finally:
    #     qn_answers = dump_data(qn_answers, d_path)
    #     return qn_answers


@get_data(load=True)
def qa_using_external_doc():
    """
    LineChain is a tool that aids in breaking down your document into smaller sections. You will create embedding vectors for each of these text segments. When a question is asked, LineChain will perform a semantic search to determine which text segment vectors closely match the question vector. It then retrieves the relevant text segment based on the question. This process ensures that the most applicable text is referred to when answering the question based on the document or text.

    There are a lot of document loaders: File Loader, Directory Loader, Notion, ReadTheDocs, HTML, PDF, PowerPoint, Email, GoogleDrive, Obsidian, Roam, EverNote, YouTube, Hacker News, GitBook, S3 File, S3 Directory, GCS File, GCS Directory, Web Base, IMSDb, AZLyrics, College Confidential, Gutenberg, Airbyte Json, CoNLL-U, iFixit, Notebook, Copypaste, CSV, Facebook Chat, Image, Markdown, SRT, Telegram, URL, Word Document, Blackboard
    """
    # https://langchain-cn.readthedocs.io/en/latest/modules/indexes/document_loaders.html
    txt_loader = TextLoader(txt1_path)
    txt_index = VectorstoreIndexCreator(
        embedding=embeddings
    ).from_loaders([txt_loader])

    res = txt_index.query(human_msg2_txt, llm=gpt3)
    return res


def execute():
    res1 = ask_questions_using_diff_provider_models()
    res2 = qa_using_external_doc()
    g = 2


if __name__ == '__main__':
    execute()
