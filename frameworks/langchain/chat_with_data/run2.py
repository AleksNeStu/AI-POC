from langchain.chains import RetrievalQA
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI

from common.cfg import *

QN = "What is the total number of AI publications between 2010 and 2022?"
pdf_loader = PyPDFLoader(str(pdf_ai_report_1))
documents = pdf_loader.load()


@get_data()
def langchain_qa():
    # oads a chain that you can use to do QA over a set of documents, but it uses ALL of those documents.
    # # chain_type="stuff" will not work because the number of tokens exceeds the limit. We can try other chain types like "map_reduce".
    llm = OpenAI(batch_size=5)
    chain = load_qa_chain(llm, chain_type="map_reduce")
    query = QN
    res = chain.run(input_documents=documents, question=query)
    return res

@get_data()
def retrieval_qa():
    """RetrievalQA chain uses load_qa_chain under the hood. We retrieve the most relevant chunk of text and feed those to the language model. """
    # - [embeddings](https://python.langchain.com/en/latest/reference/modules/embeddings.html)
    # - [TextSplitter](https://python.langchain.com/en/latest/modules/indexes/text_splitters.html)
    # - [VectorStore](https://python.langchain.com/en/latest/modules/indexes/vectorstores.html)
    # - [Retrievers](https://python.langchain.com/en/latest/modules/indexes/retrievers.html)
    #   - [search_type](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/chroma.html#mmr): "similarity" or "mmr"
    # - [Chain Type](https://python.langchain.com/en/latest/modules/chains/index_examples/question_answering.html): "stuff", "map reduce", "refine", "map_rerank"
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
    # create a chain to answer questions
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
    result = qa({"query": QN})
    return result


@get_data()
def vector_store_index_qa():
    # VectorstoreIndexCreator is a wrapper for the above logic: retrieval_qa
    index = VectorstoreIndexCreator(
        # split the documents into chunks
        text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0),
        # select which embeddings we want to use
        embedding=OpenAIEmbeddings(),
        # use Chroma as the vectorestore to index and search embeddings
        vectorstore_cls=Chroma
    ).from_loaders([pdf_loader])
    res = index.query(llm=OpenAI(), question=QN, chain_type="stuff")
    return res


@get_data()
def conversational_retrieval_chain():
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
    # create a chain to answer questions
    qa = ConversationalRetrievalChain.from_llm(OpenAI(), retriever)
    chat_history1 = []
    res1 = qa({"question": QN, "chat_history": chat_history1})
    answer1 = res1["answer"]
    chat_history2 = [(QN, res1["answer"])]
    res2 = qa({"question": QN, "chat_history": chat_history2})
    answer2 = res2["answer"]
    return (answer1, answer2)


def execute():
    res1 = langchain_qa()
    res2 = retrieval_qa()
    res3 = vector_store_index_qa()
    res4 = conversational_retrieval_chain()


if __name__ == '__main__':
    execute()