import os
import openai
import sys
from dotenv import load_dotenv, find_dotenv
import pickle
from pathlib import Path
from unittest import mock
# python -m spacy download en_core_web_md
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.document_loaders import YoutubeAudioLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import NotionDirectoryLoader

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import (
    OpenAIWhisperParser, OpenAIWhisperParserLocal,
    FasterWhisperParser, YandexSTTParser
)
import torch

from langchain_core.documents import Document

use_paid_services = False
sys.path.append('../..')
_ = load_dotenv(find_dotenv()) # read local .env file
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
# OPENAI_API_KEY = os.environ.get('API_KEY_OPEN_AI')
openai.api_key = OPENAI_API_KEY
current_dir = Path.cwd()
current_dir_parent = current_dir.parent


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# You can set compute_type to "float32" or "int8".
# Since your GPU does not support float16, you should set "int8" and not "int8_float16".
compute_type = "float16" if device == 'cuda' else 'int8'
nlp_eng = spacy.load('en_core_web_md')


result_collection = {
    'notion_data': None,
    'pdf_data': None,
    'web_data': None,
    'yt_data': None,
}

def are_texts_similar(text1, text2, use_spacy: bool = True):
    if use_spacy:
        text1_nlp = nlp_eng(text1)
        text2_nlp = nlp_eng(text2)
        text_similarity = text1_nlp.similarity(text2_nlp)

    else:
        # Vectorize texts
        tfidf_vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
        except Exception as ex:
            t = 1

        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
        text_similarity = cosine_similarities[0][0]

    return text_similarity > 0.90 # Adjust the threshold as needed


def lazy_parse_patched(self, blob):
    """Lazily parse the blob."""
    import io

    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError(
            "pydub package not found, please install it with `pip install pydub`"
        )
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster_whisper package not found, please install it with "
            "`pip install faster-whisper`"
        )

    # get the audio
    if isinstance(blob.data, bytes):
        # blob contains the audio
        audio = AudioSegment.from_file(io.BytesIO(blob.data))
    elif blob.data is None and blob.path:
        # Audio file from disk
        audio = AudioSegment.from_file(blob.path)
    else:
        raise ValueError("Unable to get audio from blob")

    file_obj = io.BytesIO(audio.export(format="mp3").read())

    # Transcribe
    # TODO: Mock FasterWhisperParser due to issue
    # https://github.com/langchain-ai/langchain/issues/23953
    model = WhisperModel(
        self.model_size, device=self.device, compute_type=compute_type
    )

    segments, info = model.transcribe(file_obj, beam_size=5)

    for segment in segments:
        yield Document(
            page_content=segment.text,
            metadata={
                "source": blob.source,
                "timestamps": "[%.2fs -> %.2fs]" % (segment.start, segment.end),
                "language": info.language,
                "probability": "%d%%" % round(info.language_probability * 100),
                **blob.metadata,
            },
        )

FasterWhisperParser.lazy_parse = lazy_parse_patched


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
def get_youtube(yt_path: Path = current_dir_parent / "data/youtube/url1.url",
                use_paid_services: bool = False,
                faster_whisper: bool = True,
                wisper_local: bool = False):
    with open(yt_path, 'r') as f:
        yt_path_str = f.read()

    tmp_dir = current_dir_parent / "tmp"
    yt_data = None
    if use_paid_services:
        yt_loader_whisper = GenericLoader(
            blob_loader=YoutubeAudioLoader([yt_path_str], str(tmp_dir)),
            blob_parser=OpenAIWhisperParser(api_key=OPENAI_API_KEY)
        )
        yt_data = yt_loader_whisper.load()
    elif faster_whisper:
        # https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.parsers.audio.FasterWhisperParser.html
        yt_loader_faster_whisper = GenericLoader(
            blob_loader=YoutubeAudioLoader([yt_path_str], str(tmp_dir)),
            blob_parser=FasterWhisperParser(device=device)
        )
        yt_data = yt_loader_faster_whisper.load()
    elif wisper_local:
        yt_loader_whisper_local = GenericLoader(
            blob_loader=YoutubeAudioLoader([yt_path_str], str(tmp_dir)),
            blob_parser=OpenAIWhisperParserLocal(device=device)
        )
        yt_data = yt_loader_whisper_local.load()
    else:
        yt_data = current_collection['yt_data']

    # Compare diff provider
    openai_res = current_collection['yt_data'][0].page_content
    local_whisper_res = yt_data[0].page_content
    assert are_texts_similar(openai_res, local_whisper_res) == True

    # print(yt_data)
    result_collection['yt_data'] = yt_data


def docs_load():
    # get_youtube(use_paid_services=False, faster_whisper=True, wisper_local=False)
    get_youtube(use_paid_services=False, faster_whisper=False, wisper_local=True)
    get_notion()
    get_pdf()
    get_web()
    # dump_collection()

def docs_split():
    g = 1

if __name__ == '__main__':
    docs_load()
    docs_split()
