import os
import openai
import sys
from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import YoutubeAudioLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import NotionDirectoryLoader

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser

from pathlib import Path



sys.path.append('../..')
_ = load_dotenv(find_dotenv()) # read local .env file

# OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_API_KEY = os.environ.get('API_KEY_OPEN_AI')
openai.api_key = OPENAI_API_KEY
current_dir = Path.cwd()
current_dir_parent = current_dir.parent



# Notion
notion_path = current_dir_parent / "data/docs/md"

notion_loader = NotionDirectoryLoader(str(notion_path))
notion_data = notion_loader.load()
notion_meta = notion_data[0].metadata
print(notion_meta)

# PDF

pdf_path = current_dir_parent / "data/docs/pdf/MachineLearning-Lecture01.pdf"
pdf_loader = PyPDFLoader(str(pdf_path))
pdf_data = pdf_loader.load()
assert len(pdf_data) == 22
page = pdf_data[3]
print(page.page_content[:100])
print(page.metadata)

# URLs (web data)
md_url = 'https://github.com/langchain-ai/langchain/blob/master/README.md'
web_loader = WebBaseLoader(md_url)
web_data = web_loader.load()
web_data_snippet = web_data[0].page_content[500:600]
print(web_data_snippet)


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
print(yt_data)
