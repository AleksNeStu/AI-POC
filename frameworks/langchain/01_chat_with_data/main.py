import os
import openai
import sys
from dotenv import load_dotenv, find_dotenv
import pickle
from pathlib import Path


from langchain_community.document_loaders import YoutubeAudioLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import NotionDirectoryLoader

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import (
    OpenAIWhisperParser, OpenAIWhisperParserLocal,
    FasterWhisperParser, YandexSTTParser
)

from langchain_core.documents import Document




use_paid_services = False
sys.path.append('../..')
_ = load_dotenv(find_dotenv()) # read local .env file
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
# OPENAI_API_KEY = os.environ.get('API_KEY_OPEN_AI')
openai.api_key = OPENAI_API_KEY
current_dir = Path.cwd()
current_dir_parent = current_dir.parent


result_collection = {
    'notion_data': None,
    'pdf_data': None,
    'web_data': None,
    'yt_data': None,
}

def dump_collection(collection: str = 'result_collection.pkl'):
    with open(collection, 'wb') as f:
        pickle.dump(result_collection, f)

def load_collection(collection: str = 'result_collection.pkl'):
    with open(collection, 'rb') as f:
        res = pickle.load(f)
    return res

current_collection = load_collection()

# Notion
def get_notion(notion_path: Path = current_dir_parent / "data/docs/md"):
    notion_loader = NotionDirectoryLoader(str(notion_path))
    notion_data = notion_loader.load()
    notion_meta = notion_data[0].metadata
    # print(notion_meta)
    result_collection['notion_data'] = notion_data
    return notion_data

# PDF
def get_pdf(pdf_path: Path = current_dir_parent / "data/docs/pdf/MachineLearning-Lecture01.pdf"):
    pdf_loader = PyPDFLoader(str(pdf_path))
    pdf_data = pdf_loader.load()
    assert len(pdf_data) == 22
    page = pdf_data[3]
    # print(page.page_content[:100])
    # print(page.metadata)
    result_collection['pdf_data'] = pdf_data
    return pdf_data

# URLs (web data)
def get_web(md_url: str = 'https://github.com/langchain-ai/langchain/blob/master/README.md'):
    web_loader = WebBaseLoader(md_url)
    web_data = web_loader.load()
    web_data_snippet = web_data[0].page_content[500:600]
    result_collection['web_data'] = web_data
    # print(web_data_snippet)
    return web_data

# YouTube
def get_youtube(yt_path: Path = current_dir_parent / "data/youtube/url1.url", use_paid_services: bool = False):
    with open(yt_path, 'r') as f:
        yt_path_str = f.read()

    tmp_dir = current_dir_parent / "tmp"
    if use_paid_services:
        yt_loader = GenericLoader(
            blob_loader=YoutubeAudioLoader([yt_path_str], str(tmp_dir)),
            blob_parser=OpenAIWhisperParser(api_key=OPENAI_API_KEY)
        )
        # yt_loader = GenericLoader(
        #     blob_loader=YoutubeAudioLoader([yt_path_str], str(tmp_dir)),
        #     blob_parser=OpenAIWhisperParserLocal()
        # )
        yt_data = yt_loader.load()
        # print(yt_data)
    else:
        yt_data = current_collection['yt_data']
    result_collection['yt_data'] = yt_data



if __name__ == '__main__':
    get_notion()
    get_pdf()
    get_web()
    get_youtube()
    dump_collection()
    res = 1
