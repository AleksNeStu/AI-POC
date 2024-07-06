import os
import openai
import sys
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFLoader
from pathlib import Path

sys.path.append('../..')

_ = load_dotenv(find_dotenv()) # read local .env file

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

current_dir = Path.cwd()
pdf_path = current_dir / "./data/docs/pdf/MachineLearning-Lecture01.pdf"

if pdf_path.exists():
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

print(pages[0])