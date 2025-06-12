import os
import openai
import sys
from langchain_openai import ChatOpenAI
import streamlit as st
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# Initialize LLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings


def loading(pdf):
    loader = PyPDFLoader(pdf)
    pages = loader.load()
    return pages

def splitting(pages):
    r_split = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200)

    chunks = r_split.split_documents(pages)
    return chunks

def embedding():
    embedding = OpenAIEmbeddings(
        model="azure-text-embedding-3-large",
        base_url=os.environ["BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"]
    )
    return embedding

from langchain_community.vectorstores import Chroma
persist_directory = 'chroma/'

def vectorstores(splits, embedding):
    vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
    
    print(vectordb._collection.count()) 
    return vectordb

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

def llm(vectordb, question):
    vectordb.max_marginal_relevance_search(question, k=10, fetch_k=1)
    llm = ChatOpenAI(
    temperature=0,
    model="azure-gpt-4o-mini",
    base_url=os.environ["BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"]
)
    base_retriever = vectordb.as_retriever(search_kwargs={'k':1})

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
    )
    compressed_docs = compression_retriever.get_relevant_documents(question)
    pretty_print_docs(compressed_docs)


def main():
    load = loading("avn-doc.pdf")
    splits = splitting(load)
    embed = embedding()  # get the embedding model
    vector_db = vectorstores(splits, embed)  # use the model in Chroma
    llm(vector_db, "High CPU utilization là gì? ")

if __name__ == "__main__":
    main()