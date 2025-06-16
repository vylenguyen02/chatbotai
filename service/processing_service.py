import os
import openai
import sys
from langchain_openai import ChatOpenAI
import bs4
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# Initialize LLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


async def loading(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages

def splitting(pages):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
    all_splits = text_splitter.split_documents(pages)
    return all_splits

from langchain_openai import OpenAIEmbeddings  

def embedding():
    embed=OpenAIEmbeddings(
    model="azure-text-embedding-3-large",  # or "text-embedding-ada-002"
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ['AZURE_OPENAI_ENDPOINT']
)   
    return embed

from langchain_chroma import Chroma
import shutil

def vectorstores(splits, embeddings):
    persist_directory = "chroma_db"

    # Delete and rebuild the DB if it already exists
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    # Create new Chroma from documents
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="example_collection",
    )
    
    return vectordb

def load_vectorstore():
    vectordb = Chroma.from_documents()


def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import AzureChatOpenAI


def llm(vectordb, question):
    vectordb.max_marginal_relevance_search(question, k=10, fetch_k=1)
    llm = ChatOpenAI(
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
        model=os.environ["AZURE_OPENAI_COMP_DEPLOYMENT_NAME"],
        api_key=os.environ["AZURE_OPENAI_API_VERSION"]
    )
    base_retriever = vectordb.as_retriever(search_kwargs={'k':1})

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
    )
    compressed_docs = compression_retriever.invoke(question)
    pretty_print_docs(compressed_docs)
 

import asyncio
async def main():
    load = await loading("avn-doc.pdf")
    splits = splitting(load)
    embed = embedding()
    vector_db = vectorstores(splits, embed)
    llm(vector_db, "What's the weather today?")

if __name__ == "__main__":
    asyncio.run(main())