# Python
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Union, Optional, List
from dotenv import load_dotenv, find_dotenv


# Helpers
from faker import Faker
import shutil

# Network
from yarl import URL

# ML
# python -m spacy download en_core_web_md
import spacy
import torch
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

# Langchain embeddings
# TODO: Test rest of the embeddings
# from langchain.embeddings import (
#     HuggingFaceEmbeddings,     # Deprecated
#     OpenAIEmbeddings,
#     OpenVINOEmbeddings,
#     SpacyEmbeddings
# )
from langchain_huggingface import HuggingFaceEmbeddings


# Langchain store
# TODO: Test rest
from langchain_community.vectorstores import Chroma


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
db_dir = current_dir_parent / 'db'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# You can set compute_type to "float32" or "int8".
# Since your GPU does not support float16, you should set "int8" and not "int8_float16".
compute_type = "float16" if device == 'cuda' else 'int8'
nlp_eng = spacy.load('en_core_web_md')

embedding = HuggingFaceEmbeddings()
fake = Faker()


@dataclass
class CollectionData:
    notions_data: CollectionDataType = field(default_factory=list)
    pdfs_data: CollectionDataType = field(default_factory=list)
    webs_data: CollectionDataType = field(default_factory=list)
    yts_data: CollectionDataType = field(default_factory=list)

collection_data = CollectionData()

txt_target, web_md_target, pdf_page_target, notion_md_target = '', '', '', ''

@dataclass
class CollectionSplit:
    notions_split: CollectionSplitType = field(default_factory=list)
    pdfs_split: CollectionSplitType = field(default_factory=list)
    webs_split: CollectionSplitType = field(default_factory=list)
    yts_split: CollectionSplitType = field(default_factory=list)


collection_split = CollectionSplit()


def clean_dir(dir_path: Path):
    shutil.rmtree(str(dir_path))


def are_texts_similar(text1, text2, use_spacy: bool = False, use_np: bool = True):
    if use_spacy:
        text1_nlp = nlp_eng(text1)
        text2_nlp = nlp_eng(text2)
        text_similarity = text1_nlp.similarity(text2_nlp)
    elif use_np:
        text1_em = embedding.embed_query(text1)
        text2_em = embedding.embed_query(text2)
        text_similarity = np.dot(text1_em, text2_em)
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

    return text_similarity > 0.99 # Adjust the threshold as needed


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
    webs_data = []
    for web_path in webs_path:
        web_data = get_web(web_path)
        webs_data.extend(web_data)

    collection_data.webs_data = webs_data
    return webs_data

# YouTube
def get_yts(yts_path: List[URL],
            use_paid_services: bool = False,
            faster_whisper: bool = True
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
    else:
        yt_loader_whisper_local = GenericLoader(
            blob_loader=YoutubeAudioLoader(yts_url, tmp_dir_str),
            blob_parser=OpenAIWhisperParserLocal(device=device)
        )
        yts_data = yt_loader_whisper_local.load()

    # print(yt_data)
    collection_data.yts_data = yts_data


def load_docs(use_saved: bool = True):
    global collection_data
    if use_saved and collection_data_saved:
        collection_data = collection_data_saved
    else:
        get_yts(
            yts_path=[
                URL('https://www.youtube.com/watch?v=1bUy-1hGZpI&ab_channel=IBMTechnology')
            ],
            use_paid_services=False, faster_whisper=False)
        get_notions()
        get_pdfs()
        get_webs(
            webs_path=[
                URL('https://github.com/langchain-ai/langchain/blob/master/README.md')
            ]
        )
        dump_collection()

def get_targets():
    global txt_target, web_md_target, pdf_page_target, notion_md_target, collection_data
    # NOTE:# OpenAIWhisperParserLocal other can produce > 1 len of chunks, adjust if needed
    txt_target = collection_data.yts_data[0].page_content
    web_md_target = collection_data.webs_data[0].page_content
    pdf_page_target = collection_data.pdfs_data[0].page_content
    notion_md_target = collection_data.notions_data[0].page_content

def split_targets():
    global collection_data
    text_char_splitter = CharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=0,
        separator = ' '
    )
    txt_split_char = text_char_splitter.split_text(txt_target)
    #assert len(txt_split_char) == 19

    text_rec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=0,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )
    txt_split_rec = text_rec_splitter.split_text(txt_target)
    # assert len(txt_split_rec) == 58

    text_docf_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    pdf_split_docf = text_docf_splitter.split_documents(
        collection_data.pdfs_data)
    # assert len(collection_data.pdfs_data) == 22
    # assert len(pdf_split_docf) == 77

    notion_split_docf = text_docf_splitter.split_documents(
        collection_data.notions_data)
    # assert len(collection_data.notions_data) == 34
    # assert len(notion_split_docf) == 442

    token_splitter_docf = TokenTextSplitter(chunk_size=20, chunk_overlap=0)
    pdf_token_split_docf = token_splitter_docf.split_documents(
        collection_data.pdfs_data)
    # assert len(txt_split_rec) == 58

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    notion_md_targets_txt = ' '.join([d.page_content for d in collection_data.notions_data])
    md_header_splits = markdown_splitter.split_text(notion_md_targets_txt)


def store_data(documents: List[Document], db_dir: Path = db_dir, to_clean_dir: bool = False):
    db_dir_str = str(db_dir)
    vector_db = None
    # Save to db
    if to_clean_dir:
        clean_dir(db_dir)
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=db_dir_str
        )

    # ValueError: You must provide an embedding function to compute embeddings.https://docs.trychroma.com/guides/embeddings
    vector_db = Chroma(persist_directory=db_dir_str, embedding_function=embedding)
    assert vector_db._collection.count() == len(documents)
    # LangChainDeprecationWarning: Since Chroma 0.4.x the manual persistence method is no longer supported as docs are automatically persisted.
    # vector_db.persist()

    return vector_db



def similarity_search(vector_db, qn: str, k = 5):
    # Store
    # Normal query
    docs_query = vector_db.similarity_search(query=qn, k)
    query_content = [doc_qr.page_content for doc_qr in docs_query]
    query_metadata = [doc_qr.metadata for doc_qr in docs_query]
    assert len(query_content) == 5
    # assert all([('email' in qc) for qc in query_content]) == Tru
    return query_metadata, query_content



def embedding_data():
    # An embedding function is a function that converts your data (in this case, probably text data) into a numerical vector representation that can be used for similarity comparisons. This is a key part of many machine learning and information retrieval systems.
    fake_left = fake.text(max_nb_chars=199)
    fake_right = fake.text(max_nb_chars=199)
    txt_target_fake = fake_left + txt_target + fake_right
    sim_texts = are_texts_similar(txt_target, txt_target_fake)
    assert sim_texts == False

    txt_target_em = embedding.embed_query(txt_target)
    web_md_target_em = embedding.embed_query(web_md_target)
    pdf_page_target_em = embedding.embed_query(pdf_page_target)
    notion_md_target_em = embedding.embed_query(notion_md_target)

    # Split docs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 150
    )
    pdfs_splits = text_splitter.split_documents(collection_data.pdfs_data)
    vector_db = store_data(documents=pdfs_splits)

    # Got relevant and distinct results  due to data duplication by fake data
    meta_1, content_1 = similarity_search(
        vector_db, qn="is there an email i can ask for help")
    assert content_1[0] == content_1[1]

    # Edge cases (duplications)
    meta_2, content_2 = similarity_search(
        vector_db, qn="what did they say about regression in the third lecture?")
    assert content_1[0] == content_1[1]
    assert len(set(meta_2)) < len(meta_1)



def add_dirty_data():
    extra_pdf_data = get_pdf()
    collection_data.pdfs_data.extend(extra_pdf_data)


if __name__ == '__main__':
    load_docs(use_saved=True)
    get_targets()
    split_targets()
    add_dirty_data()
    embedding_data()
