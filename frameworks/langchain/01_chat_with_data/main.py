# Python
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Union, Optional, List
from dotenv import load_dotenv, find_dotenv

# Network
from yarl import URL

# ML
# python -m spacy download en_core_web_md
import spacy
import torch
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Langchain Document loaders
from langchain_community.document_loaders import NotionDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import YoutubeAudioLoader
from langchain_community.document_loaders.generic import GenericLoader

# Langchain Document loaders Audio
from langchain_community.document_loaders.parsers.audio import (
    OpenAIWhisperParser, OpenAIWhisperParserLocal,
    FasterWhisperParser, YandexSTTParser
)

# Langchain text splitter
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    #TODO: Test rest of the splitters
    # MarkdownTextSplitter,
    # SentenceTransformersTokenTextSplitter,
    # Language,
    # NLTKTextSplitter,
    # SpacyTextSplitter,
)

# Langchain Document loaders help
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_core.documents import Document


CollectionDataType = Optional[Iterator[Document]]
CollectionSplitType = Optional[Union[Iterator[Document], str]]

use_paid_services = False
sys.path.append('../..')
_ = load_dotenv(find_dotenv()) # read local .env file
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
# OPENAI_API_KEY = os.environ.get('API_KEY_OPEN_AI')
openai.api_key = OPENAI_API_KEY
current_dir = Path.cwd()
current_dir_parent = current_dir.parent
md_dir = current_dir_parent / 'data/docs/md'
pdf_dir = current_dir_parent / 'data/docs/pdf/'
tmp_dir = current_dir_parent / "tmp"


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# You can set compute_type to "float32" or "int8".
# Since your GPU does not support float16, you should set "int8" and not "int8_float16".
compute_type = "float16" if device == 'cuda' else 'int8'
nlp_eng = spacy.load('en_core_web_md')


@dataclass
class CollectionData:
    notions_data: CollectionDataType = None
    pdfs_data: CollectionDataType = None
    webs_data: CollectionDataType = None
    yts_data: CollectionDataType = None

collection_data = CollectionData()

txt_target, web_md_target, pdf_page_target, notion_md_target = '', '', '', ''

@dataclass
class CollectionSplit:
    notions_split: CollectionSplitType = None
    pdfs_split: CollectionSplitType = None
    webs_split: CollectionSplitType = None
    yts_split: CollectionSplitType = None


collection_split = CollectionSplit()


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


def lazy_parse_patched(self, blob: Blob) -> Iterator[Document]:
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


def dump_collection(collection: str = 'collection_data'):
    collection_file = f'{collection}.pkl'
    with open(collection_file, 'wb') as f:
        pickle.dump(collection_data, f)

def load_collection(collection: str = 'collection_data'):
    collection_file = f'{collection}.pkl'
    with open(collection_file, 'rb') as f:
        res = pickle.load(f)
    return res


collection_data_saved = load_collection('collection_data') if (current_dir / 'collection_data.pkl').exists() else None
collection_split_saved = load_collection('collection_split') if (current_dir / 'collection_split.pkl').exists() else None

# Notion
def get_notions(notions_path: Path = md_dir):
    notions_loader = NotionDirectoryLoader(str(notions_path))
    notions_data = notions_loader.load()
    # print(notions_data[0].metadata)
    collection_data.notions_data = notions_data
    return notions_data

# PDF
def get_pdf(pdf_path: Path = pdf_dir / "MachineLearning-Lecture01.pdf"):
    pdf_loader = PyPDFLoader(str(pdf_path))
    pdf_data = pdf_loader.load()
    # page = pdf_data[3]
    # print(page.page_content[:100])
    # print(page.metadata)
    return pdf_data

def get_pdfs(pdfs_path: Path = pdf_dir):
    pdfs_loader = PyPDFDirectoryLoader(str(pdfs_path))
    pdfs_data = pdfs_loader.load()
    collection_data.pdfs_data = pdfs_data
    return pdfs_data

# URLs (web data)
def get_web(web_url: URL = 'https://github.com/langchain-ai/langchain/blob/master/README.md'):
    web_loader = WebBaseLoader(str(web_url))
    web_data = web_loader.load()
    # print( web_data[0].page_content[500:600])
    return web_data

def get_webs(webs_path: List[URL]):
    webs_data = [get_web(web_path) for web_path in webs_path]
    collection_data.webs_data = webs_data
    return webs_data

# YouTube
def get_yts(yts_path: List[URL],
            use_paid_services: bool = False,
            faster_whisper: bool = True,
            wisper_local: bool = False
            ):
    # with open(yt_path, 'r') as f:
    #     yt_path_str = f.read()
    yts_data = None
    yts_url = [str(yt_path) for yt_path in yts_path]
    tmp_dir_str = str(tmp_dir)
    if use_paid_services:
        yt_loader_whisper = GenericLoader(
            blob_loader=YoutubeAudioLoader(yts_url, tmp_dir_str),
            blob_parser=OpenAIWhisperParser(api_key=OPENAI_API_KEY)
        )
        blob_parser = YandexSTTParser  #TODO: Test and extend the function
        yts_data = yt_loader_whisper.load()
    elif faster_whisper:
        # https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.parsers.audio.FasterWhisperParser.html
        yt_loader_faster_whisper = GenericLoader(
            blob_loader=YoutubeAudioLoader(yts_url, tmp_dir_str),
            blob_parser=FasterWhisperParser(device=device)
        )
        yts_data = yt_loader_faster_whisper.load()
    elif wisper_local:
        yt_loader_whisper_local = GenericLoader(
            blob_loader=YoutubeAudioLoader(yts_url, tmp_dir_str),
            blob_parser=OpenAIWhisperParserLocal(device=device)
        )
        yts_data = yt_loader_whisper_local.load()
    else:
        yts_data = collection_data_saved.yt_data

    if collection_data_saved:
        # Compare diff provider
        openai_res = collection_data_saved.yt_data[0].page_content
        local_whisper_res = yts_data[0].page_content
        assert are_texts_similar(openai_res, local_whisper_res) == True

    # print(yt_data)
    collection_data.yts_data = yts_data


def docs_load():
    get_yts(
        yts_path=[
            URL('https://www.youtube.com/watch?v=1bUy-1hGZpI&ab_channel=IBMTechnology')
        ],
        use_paid_services=False, faster_whisper=False, wisper_local=True)
    get_notions()
    get_pdfs()
    get_webs(
        webs_path=[
            URL('https://github.com/langchain-ai/langchain/blob/master/README.md')
        ]
    )
    dump_collection()

def get_targets():
    global txt_target, web_md_target, pdf_page_target, notion_md_target
    if not collection_data_saved:
        docs_load()
    # NOTE:# OpenAIWhisperParserLocal other can produce > 1 len of chunks, adjust if needed
    assert len(collection_data_saved.yt_data) == 1
    txt_target = collection_data_saved.yt_data[0].page_content

    assert len(collection_data_saved.web_data) == 1
    web_md_target = collection_data_saved.web_data[0].page_content

    assert len(collection_data_saved.pdf_data) == 22
    pdf_page_target = collection_data_saved.pdf_data[7].page_content

    assert len(collection_data_saved.notion_data) == 34
    notion_md_target = collection_data_saved.notion_data[7].page_content

def docs_split():
    get_targets()
    text_char_splitter = CharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=0,
        separator = ' '
    )
    txt_split_char = text_char_splitter.split_text(txt_target)
    assert len(txt_split_char) == 19

    text_rec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=0,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )
    txt_split_rec = text_rec_splitter.split_text(txt_target)
    assert len(txt_split_rec) == 58

    text_docf_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    pdf_split_docf = text_docf_splitter.split_documents(
        collection_data_saved.pdf_data)
    assert len(collection_data_saved.pdf_data) == 22
    assert len(pdf_split_docf) == 77

    notion_split_docf = text_docf_splitter.split_documents(
        collection_data_saved.notion_data)
    assert len(collection_data_saved.notion_data) == 34
    assert len(notion_split_docf) == 442

    token_splitter_docf = TokenTextSplitter(chunk_size=20, chunk_overlap=0)
    pdf_token_split_docf = token_splitter_docf.split_documents(
        collection_data_saved.pdf_data)
    assert len(txt_split_rec) == 58

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    notion_md_targets_txt = ' '.join([d.page_content for d in collection_data_saved.notion_data])
    md_header_splits = markdown_splitter.split_text(notion_md_targets_txt)


if __name__ == '__main__':
    docs_load()
    docs_split()
