import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Union, Optional, List
from uuid import uuid4

import numpy as np
import openai
# python -m spacy download en_core_web_md
import spacy
from faker import Faker
import panel as pn  # GUI
from langchain.chains import RetrievalQA
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.globals import set_debug
from langchain.prompts import PromptTemplate
from langchain.retrievers import (
    SelfQueryRetriever, ContextualCompressionRetriever)
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    # TODO: Test rest of the splitters
    # MarkdownTextSplitter,
    # SentenceTransformersTokenTextSplitter,
    # Language,
    # NLTKTextSplitter,
    # SpacyTextSplitter,
)
from langchain_community.document_loaders import NotionDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import YoutubeAudioLoader
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.generic import GenericLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders.parsers.audio import (
    OpenAIWhisperParser,
    OpenAIWhisperParserLocal,
    FasterWhisperParser,
    YandexSTTParser,
)
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import SVMRetriever, TFIDFRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_openai import ChatOpenAI
# from langchain.llms import OpenAI
from langchain_openai import OpenAI
# TODO: Test rest of the embeddings
# from langchain.embeddings import (
#     HuggingFaceEmbeddings,     # Deprecated
#     OpenAIEmbeddings,
#     OpenVINOEmbeddings,
#     SpacyEmbeddings
# )
from langchain_openai import OpenAIEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from yarl import URL

from common import ml
from common.cfg import *


CollectionDataType = Optional[Iterator[Document]]
CollectionSplitType = Optional[Union[Iterator[Document], str]]

use_paid_services = True


openai.api_key = OPENAI_API_KEY
current_dir = Path(__file__).resolve().parent

collection_path = current_dir / 'collection_data.pkl'
collection_split_path = current_dir / 'collection_split.pkl'


tmp_dir = root_dir / "tmp"

db_dir = current_dir / "db"


PROMPT_TMPL = ("""
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try 
to make up an answer.
Use two sentences maximum, one jf them from wikipedia.
Keep the answer as concise as possible.
Always say "thanks bro!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:
""")


nlp_eng = spacy.load("en_core_web_md")


fake = Faker()

if use_paid_services:
    embedding = OpenAIEmbeddings()
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
    llm_chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
else:
    embedding = HuggingFaceEmbeddings()
    # embedding = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
    # llm = ml.get_llm_distilgpt2()
    llm = ml.get_llm_gpt2_medium()
    llm_chat = ml.get_llm_chat_hf(llm)
    compressor = LLMChainExtractor.from_llm(llm)
    # llm = ml.hugging_face_model


@dataclass
class CollectionData:
    notions_data: CollectionDataType = field(default_factory=list)
    pdfs_data: CollectionDataType = field(default_factory=list)
    webs_data: CollectionDataType = field(default_factory=list)
    yts_data: CollectionDataType = field(default_factory=list)


collection_data = CollectionData()

txt_target, web_md_target, pdf_page_target, notion_md_target = "", "", "", ""


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

    return text_similarity > 0.99  # Adjust the threshold as needed


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
    model = WhisperModel(self.model_size, device=self.device, compute_type=compute_type)

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


collection_data_saved = (
    load_data(collection_path)
    if collection_path.exists()
    else None
)
collection_split_saved = (
    load_data(collection_split_path)
    if collection_split_path.exists()
    else None
)


# Notion
def get_notions(notions_path: Path = md_dir):
    notions_loader = NotionDirectoryLoader(str(notions_path))
    notions_data = notions_loader.load()
    # print(notions_data[0].metadata)
    collection_data.notions_data = notions_data
    return notions_data

def get_db_doc_count(vector_db, to_print = True):
    res = vector_db._collection.count()
    if to_print:
        print(f"DB collection count: {res}")
    return res

# PDF
def get_pdf(pdf_path: Path = pdf_1_path):
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
def get_web(
    web_url: URL = "https://github.com/langchain-ai/langchain/blob/master/README.md",
):
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
def get_yts(
    yts_path: List[URL], use_paid_services: bool = False, faster_whisper: bool = True
):
    # with open(yt_path, 'r') as f:
    #     yt_path_str = f.read()
    yts_data = None
    yts_url = [str(yt_path) for yt_path in yts_path]
    tmp_dir_str = str(tmp_dir)
    if use_paid_services:
        yt_loader_whisper = GenericLoader(
            blob_loader=YoutubeAudioLoader(yts_url, tmp_dir_str),
            blob_parser=OpenAIWhisperParser(api_key=OPENAI_API_KEY),
        )
        blob_parser = YandexSTTParser  # TODO: Test and extend the function
        yts_data = yt_loader_whisper.load()
    elif faster_whisper:
        # https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.parsers.audio.FasterWhisperParser.html
        yt_loader_faster_whisper = GenericLoader(
            blob_loader=YoutubeAudioLoader(yts_url, tmp_dir_str),
            blob_parser=FasterWhisperParser(device=device),
        )
        yts_data = yt_loader_faster_whisper.load()
    else:
        yt_loader_whisper_local = GenericLoader(
            blob_loader=YoutubeAudioLoader(yts_url, tmp_dir_str),
            blob_parser=OpenAIWhisperParserLocal(device=device),
        )
        yts_data = yt_loader_whisper_local.load()

    # print(yt_data)
    collection_data.yts_data = yts_data


def load_docs(use_saved: bool = True, add_duty_data: bool = False):
    global collection_data
    if use_saved and collection_data_saved:
        collection_data = collection_data_saved
    else:
        get_yts(
            yts_path=[
                URL(
                    "https://www.youtube.com/watch?v=1bUy-1hGZpI&ab_channel=IBMTechnology"
                )
            ],
            use_paid_services=False,
            faster_whisper=False,
        )
        get_notions()
        get_pdfs()
        get_webs(
            webs_path=[
                URL("https://github.com/langchain-ai/langchain/blob/master/README.md")
            ]
        )
        dump_data(collection_data, collection_path)

        add_dirty_data_to_collection()


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
        chunk_size=450, chunk_overlap=0, separator=" "
    )
    txt_split_char = text_char_splitter.split_text(txt_target)
    # assert len(txt_split_char) == 19

    text_rec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150, chunk_overlap=0, separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )
    txt_split_rec = text_rec_splitter.split_text(txt_target)
    # assert len(txt_split_rec) == 58

    text_docf_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=150, length_function=len
    )
    pdf_split_docf = text_docf_splitter.split_documents(collection_data.pdfs_data)
    # assert len(collection_data.pdfs_data) == 22
    # assert len(pdf_split_docf) == 77

    notion_split_docf = text_docf_splitter.split_documents(collection_data.notions_data)
    # assert len(collection_data.notions_data) == 34
    # assert len(notion_split_docf) == 442

    token_splitter_docf = TokenTextSplitter(chunk_size=20, chunk_overlap=0)
    pdf_token_split_docf = token_splitter_docf.split_documents(
        collection_data.pdfs_data
    )
    # assert len(txt_split_rec) == 58

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    notion_md_targets_txt = " ".join(
        [d.page_content for d in collection_data.notions_data]
    )
    md_header_splits = markdown_splitter.split_text(notion_md_targets_txt)


def get_db(db_dir: Path | None = db_dir, name: str = 'langchain'):
    if db_dir:
        db_dir_str = str(db_dir)
        if db_dir.exists() and (db_dir / "chroma.sqlite3").exists():
            vector_db = Chroma(persist_directory=db_dir_str, embedding_function=embedding)
            return vector_db
    mem_vector_db = Chroma(embedding_function=embedding)
    return mem_vector_db


def create_db(
        name:str, db_dir: Path | None = db_dir, to_clean_dir: bool = False, documents: List[Document] = None
):
    # Save to db
    vector_db = None
    db_dir_str = str(db_dir)
    if to_clean_dir and db_dir.exists():
        clean_dir(db_dir)
        del vector_db

    if documents:
        vector_db = Chroma.from_documents(
            embedding=embedding, persist_directory=db_dir_str, documents=documents,
            collection_name=name
        )
    else:
        vector_db = Chroma(persist_directory=db_dir_str, embedding_function=embedding, collection_name=name)

    return vector_db


def similarity_search(vector_db, qn: str, k=5):
    # Store
    # Normal query
    docs_query = vector_db.similarity_search(query=qn, k=k)
    query_content = [doc_qr.page_content for doc_qr in docs_query]
    query_metadata = [doc_qr.metadata for doc_qr in docs_query]
    # assert len(query_content) == 5
    # assert all([('email' in qc) for qc in query_content]) == Tru
    return query_metadata, query_content


def max_marginal_relevance_search(vector_db, qn: str, k=5, fetch_k=3):
    # Store
    # Normal query
    docs_query = vector_db.max_marginal_relevance_search(query=qn, k=k, fetch_k=fetch_k)
    query_content = [doc_qr.page_content for doc_qr in docs_query]
    query_metadata = [doc_qr.metadata for doc_qr in docs_query]
    # assert len(query_content) == 5
    # assert all([('email' in qc) for qc in query_content]) == Tru
    return query_metadata, query_content


# TODO: Test https://github.com/chroma-core/chroma
def init_db(name: str = 'langchain', documents: List[Document] = None, to_clean_dir: bool = False):
    vector_db = get_db(db_dir, name)
    if not vector_db or to_clean_dir:
        vector_db = create_db(name=name, documents=documents, to_clean_dir=to_clean_dir)

    return vector_db

def init_mem_db_w_pdf(pdf_path: Path = pdf_langchain):
    in_memory_db = get_db(db_dir=None, name='in_memory')
    pdf_f_docs = get_pdf(pdf_path)
    get_db_doc_count(in_memory_db)
    add_to_db(in_memory_db, pdf_f_docs)
    get_db_doc_count(in_memory_db)
    return in_memory_db

def add_to_db(vector_db: Chroma, documents: List[Document] = None, texts: List[str] = None):
    if documents:
        vector_db.add_documents(documents)
    if texts:
        vector_db.add_texts(texts)
    return vector_db



class TestMLCases:

    @staticmethod
    def test_mmr(vector_db):
        # Similarity Search

        # Addressing Diversity: Maximum marginal relevance
        # How to enforce diversity in the search results.
        qn_1 = "Tell me about all-white mushrooms with large fruiting bodies"
        docs_1 = vector_db.similarity_search(qn_1, k=2)
        docs_mmr_1 = vector_db.max_marginal_relevance_search(qn_1, k=2, fetch_k=3)

        qn_2 = "what did they say about matlab?"
        docs_2 = vector_db.similarity_search(qn_2, k=3)
        docs_mmr_2 = vector_db.max_marginal_relevance_search(qn_2, k=3)
        # MMR is a method used to avoid redundancy while retrieving relevant items to a query. Instead of merely retrieving the most relevant items (which can often be very similar to each other), MMR ensures a balance between relevancy and diversity in the items retrieved.


    @staticmethod
    def test_filter_and_self_query_retriever(vector_db, run_qr: bool = False):
        global llm
        # Working with metadata
        # LLM Aided Retrieval (Working with metadata)
        qn_3 = "what did they say about regression in the third lecture?"
        src_val = str(pdf_3_path)
        docs_filter_3_1 = vector_db.similarity_search(qn_3, k=3, filter={"source": src_val})
        docs_filter_3_2 = vector_db.similarity_search(qn_3, k=3, filter={"page": 7})

        # Working with metadata using self-query retriever
        # TODO: Test other LLMs
        # Time consuming op on CPU and laptops
        if run_qr:
            # Compression LLM: Increase the number of results you can put in the
            # context by shrinking the responses to only the relevant Information.
            # Working with metadata using self-query retriever
            # To address this, we can use SelfQueryRetriever, which uses an LLM to extract:
            # The query string to use for vector search
            # A metadata filter to pass in as well
            metadata_field_info = [
                AttributeInfo(
                    name="source",
                    description=(
                        f"The lecture the chunk is from, should be one of "
                        f"`{str(pdf_1_path)}`, `{str(pdf_2_path)}`, or `{str(pdf_3_path)}`"
                    ),
                    type="string",
                ),
                AttributeInfo(
                    name="page",
                    description="The page from the lecture",
                    type="integer",
                ),
            ]
            # **Note:** The default model for `OpenAI` ("from langchain.llms import OpenAI") is `text-davinci-003`.
            # Due to the deprication of OpenAI's model `text-davinci-003` on 4 January 2024, you'll be using OpenAI's
            # recommended replacement model `gpt-3.5-turbo-instruct` instead.
            document_content_description = "Lecture notes"

            # """Retriever that uses a vector store and an LLM to generate the vector store queries."""

            # tokenizer = AutoTokenizer.from_pretrained(model_name)

            # https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/self_query/
            self_query_retriever = SelfQueryRetriever.from_llm(
                llm=llm,
                vectorstore=vector_db,
                document_contents=document_content_description,
                metadata_field_info=metadata_field_info,
                verbose=True,
                k=1,
                enable_limit=True
            )
            # ValueError: Input length of input_ids is 1331, but `max_length` is set to 50. This can lead to unexpected behavior. You should consider increasing `max_length` or, better yet, setting `max_new_tokens`.

            # This happens because the input_ids (tokenized representation of the input text) have a length of 1331,
            # which is beyond the model's capacity.
            qn_4 = "what did they say about regression in the third lecture?"
            qn_4 = "what did"

            db_docs = vector_db.get().get("documents")
            len_db_docs = len(db_docs) if db_docs else 0
            docs_4 = self_query_retriever.get_relevant_documents(qn_4)
            # docs_4 = self_query_retriever.invoke(qn_4)
            pretty_print_docs(docs_4)

    @staticmethod
    def test_get_and_split_targets(vector_db):
        get_targets()
        split_targets()

    @staticmethod
    def test_contextual_compression_retriever(vector_db):
        # Another approach for improving the quality of retrieved docs is compression.
        # Information most relevant to a query may be buried in a document with a lot of irrelevant text.
        # Passing that full document through your application can lead to more expensive LLM calls and poorer responses.
        # Contextual compression is meant to fix this.
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=vector_db.as_retriever()
        )
        qn = "what did they say about matlab?"
        max_length = len(vector_db.get().get("documents")) + 100
        compressed_docs = compression_retriever.invoke(qn)
        pretty_print_docs(compressed_docs)


    @staticmethod
    def test_mix_mmr_and_compression(vector_db):
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vector_db.as_retriever(search_type="mmr")
        )
        question = "what did they say about matlab?"
        compressed_docs = compression_retriever.get_relevant_documents(question)
        pretty_print_docs(compressed_docs)


    @staticmethod
    def test_other_retrievers(vector_db):
        # Vectordb as not the only kind of tool to retrieve documents.
        # We have also TF-IDF or SVM.
        pages = collection_data.pdfs_data
        all_page_text=[p.page_content for p in pages]
        joined_page_text=" ".join(all_page_text)
        # Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500,chunk_overlap = 150)
        splits = text_splitter.split_text(joined_page_text)
        # Retrieve
        svm_retriever = SVMRetriever.from_texts(splits, embedding)
        tfidf_retriever = TFIDFRetriever.from_texts(splits)

        qn1 = "What are major topics for this class?"
        docs_svm = svm_retriever.get_relevant_documents(qn1)
        res1 = docs_svm[0]

        qn2 = "what did they say about matlab?"
        docs_tfidf=tfidf_retriever.get_relevant_documents(qn2)
        res2 = docs_tfidf[0]

    @staticmethod
    def test_embedding_data(vector_db):
        # An embedding function is a function that converts your data (in this case, probably text data) into a numerical vector representation that can be used for similarity comparisons. This is a key part of many machine learning and information retrieval systems.
        txt_target_em = embedding.embed_query(txt_target)
        web_md_target_em = embedding.embed_query(web_md_target)
        pdf_page_target_em = embedding.embed_query(pdf_page_target)
        notion_md_target_em = embedding.embed_query(notion_md_target)

    @staticmethod
    def test_similarity(vector_db):
        fake_left = fake.text(max_nb_chars=199)
        fake_right = fake.text(max_nb_chars=199)
        txt_target_fake = fake_left + txt_target + fake_right
        sim_texts = are_texts_similar(txt_target, txt_target_fake)
        assert sim_texts == False

    @staticmethod
    def test_got_distinct_res(vector_db):
        # Got relevant and distinct results  due to data duplication by fake data
        meta_1, content_1 = similarity_search(
            vector_db, qn="is there an email i can ask for help"
        )
        # assert content_1[0] == content_1[1]
        # Edge cases (duplications)
        meta_2, content_2 = similarity_search(
            vector_db, qn="what did they say about regression in the third lecture?"
        )
        # assert content_1[0] == content_1[1]
        # assert len(set(meta_2)) < len(meta_1)

    @staticmethod
    def test_qa(vector_db, load_res = False, chain_type = 'stuff'):
        t_name = TestMLCases.test_qa.__name__
        data_path = root_dir / f'{t_name}.pkl'
        if load_res:
            res = load_data(data_path)
            return res

        qa_chain = RetrievalQA.from_chain_type(
            llm_chat,
            chain_type=chain_type,
            retriever=vector_db.as_retriever())
        qn = "What is a topic?"
        res = qa_chain({"query": qn})
        res_answer = res['result']
        print(f'{t_name} answer: {res_answer}')
        # for pdf_langchain
        is_sim = are_texts_similar(
            res_answer, "A topic can refer to a subject or theme that is being discussed, studied, "
                                "or written about. In the context of LangChain, a topic could be related to the subject matter or focus of the applications being developed using the framework.")
        # assert is_sim == True
        # assert res == (
        #     "The topics discussed in the context provided include locally weighted regression, linear regression, logistic regression, the perceptron algorithm, and Newton's method."
        # )
        dump_data(res, data_path)
        return res

    @staticmethod
    def test_debug_qa_diff_chain_types(
            vector_db, load_res = False, chain_type ='stuff'):
        # console
        t_name = TestMLCases.test_debug_qa_diff_chain_types.__name__
        data_path = root_dir / f'{t_name}.pkl'
        if load_res:
            res = load_data(data_path)
            return res


        # remote web tool
        # https://api.python.langchain.com/en/latest/tracers/langchain_core.tracers.context.tracing_v2_enabled.html
        # https://www.langchain.com/langsmith)
        # https://python.langchain.com/v0.1/docs/langsmith/walkthrough/
        unique_id = uuid4().hex[0:8]
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = f"Tracing {t_name} - {unique_id}"
        # os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        set_debug(True)

        with tracing_v2_enabled(project_name="My Project"):
            results = []
            qn = "Is probability a class topic?"
            qa_chain_mr = RetrievalQA.from_chain_type(
                llm,
                retriever=vector_db.as_retriever(),
                chain_type="map_reduce"
            )
            res_mr = qa_chain_mr({"query": qn})
            res_answer_mr = res_mr["result"]
            results.append(res_mr)

            qa_chain_refine = RetrievalQA.from_chain_type(
                llm,
                retriever=vector_db.as_retriever(),
                chain_type="refine"
            )
            res_refine = qa_chain_refine({"query": qn})
            res_answer_refine = res_refine["result"]
            results.append(res_refine)

            dump_data(results, data_path)
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            set_debug(False)
            return results

    @staticmethod
    def test_debug_qa_w_history(
            vector_db, load_res = False, memory_key ='chat_history'):
        pn.extension()
        memory = ConversationBufferMemory(
            memory_key=memory_key,
            return_messages=True
        )
        retriever=vector_db.as_retriever()
        qa = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            memory=memory
        )
        qn1 = "Is probability a class topic?"
        res1 = qa({"question": qn1})
        res1_answer = res1['answer']
        qn2 = "why are those prerequesites needed?"
        res2 = qa({"question": qn2})
        res2_answer = res2['answer']






    @staticmethod
    def test_qa_w_prompt_tmpl(vector_db, load_res = False, chain_type = 'stuff'):
        t_name = TestMLCases.test_qa_w_prompt_tmpl.__name__
        data_path = root_dir / f'{t_name}.pkl'
        if load_res:
            res = load_data(data_path)
            return res

        qa_chain_prompt = PromptTemplate.from_template(PROMPT_TMPL)
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vector_db.as_retriever(),
            chain_type=chain_type,
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_chain_prompt}
        )
        question = "Is probability a class topic?"
        res = qa_chain({"query": question})
        res_answer = res['result']
        print(f'{t_name} answer: {res_answer}')
        res_docs_src = res['source_documents']
        dump_data(res, data_path)
        return res



def pretty_print_docs(docs):
    res = (
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )
    # print(res)
    return res


def add_to_db_pdf(vector_db, to_split: bool = False):
    if to_split:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=150
        )
        documents = text_splitter.split_documents(collection_data.pdfs_data)
        add_to_db(vector_db, documents=documents)
    else:
        add_to_db(vector_db, documents=collection_data.pdfs_data)


def add_to_db_fake_texts(vector_db):
    fake_texts = fake.texts(200)
    vector_db = add_to_db(vector_db, texts=fake_texts)


def add_to_db_mmr_text(vector_db):
    texts = [
        """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
        """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
        """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
    ]
    add_to_db(vector_db, texts=texts)


def run_tests_1(vector_db, vector_db_mem, to_clean_dir: bool = False):
    # TestMLCases.test_get_and_split_targets(vector_db)  # test
    # TestMLCases.test_similarity(vector_db)
    # TestMLCases.test_embedding_data(vector_db)
    # TestMLCases.test_got_distinct_res(vector_db)
    # TestMLCases.test_mmr(vector_db)
    # TestMLCases.test_filter_and_self_query_retriever(vector_db, run_qr=True)
    # TestMLCases.test_contextual_compression_retriever(vector_db)
    # TestMLCases.test_mix_mmr_and_compression(vector_db)
    # TestMLCases.test_other_retrievers(vector_db)

    res_10 = TestMLCases.test_qa(vector_db_mem, load_res=True)
    # map_reduce more solve cause answer each qn individually for collection many docs
    # res_11 = TestMLCases.test_qa(vector_db_mem, load_res=True, chain_type='map_reduce')
    res_12 = TestMLCases.test_qa_w_prompt_tmpl(vector_db_mem, load_res=True)
    res_13 = TestMLCases.test_debug_qa_diff_chain_types(vector_db_mem, load_res=True)
    res_14 = TestMLCases.test_debug_qa_w_history(vector_db_mem, load_res=True)

def add_to_db_batch_2(vector_db):
    g = 1


def run_tests_2(vector_db):
    pass


def add_dirty_data_to_collection():
    extra_pdf_data = get_pdf()
    collection_data.pdfs_data.extend(extra_pdf_data)


def add_to_db_batch_1(vector_db, pdf_only: bool = False, run: bool = True):
    if run:
        get_db_doc_count(vector_db)
        add_to_db_pdf(vector_db)
        if not pdf_only:
            add_to_db_fake_texts(vector_db)
            add_to_db_mmr_text(vector_db)

def execute(tests_1 = True, tests_2 = True):
    load_docs(use_saved=True, add_duty_data=True)
    vector_db_mem = init_mem_db_w_pdf()

    if tests_1:
        vector_db_1 = init_db(name='tests_1', to_clean_dir=False)
        # run True 1 time only
        add_to_db_batch_1(vector_db_1, pdf_only=True, run=False)
        get_db_doc_count(vector_db_1)
        run_tests_1(vector_db_1, vector_db_mem, to_clean_dir=False)
    elif tests_2:
        vector_db_2 = init_db(name='tests_2', to_clean_dir=False)
        run_tests_2(vector_db=vector_db_2)

if __name__ == "__main__":
    execute()