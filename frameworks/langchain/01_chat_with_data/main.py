import os
import openai
import sys
from dotenv import load_dotenv, find_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders import WebBaseLoader

from pathlib import Path
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser


sys.path.append('../..')
_ = load_dotenv(find_dotenv()) # read local .env file

# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_API_KEY = os.environ.get('API_KEY_OPEN_AI')
openai.api_key = OPENAI_API_KEY
current_dir = Path.cwd()
current_dir_parent = current_dir.parent

# PDF

pdf_path = current_dir_parent / "data/docs/pdf/MachineLearning-Lecture01.pdf"
pdf_loader = PyPDFLoader(str(pdf_path))
pdf_data = pdf_loader.load()
assert len(pdf_data) == 22

# YouTube
yt_path = current_dir_parent / "data/youtube/url1.url"
with open(yt_path, 'r') as f:
    yt_path_str = f.read()

tmp_dir = current_dir_parent / "tmp"
yt_loader = GenericLoader(
    blob_loader=YoutubeAudioLoader([yt_path_str], str(tmp_dir)),
    blob_parser=OpenAIWhisperParser()
)
yt_data = yt_loader.load()

# URLs (web data)
md_url = 'https://github.com/langchain-ai/langchain/blob/master/README.md'
web_loader = WebBaseLoader(md_url)
web_data = web_loader.load()
web_data_snippet = web_data[0].page_content[500:]