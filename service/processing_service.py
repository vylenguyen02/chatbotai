import os
import openai
import sys
from langchain_openai import ChatOpenAI
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

llm = ChatOpenAI(
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
        model=os.environ["AZURE_OPENAI_COMP_DEPLOYMENT_NAME"],
        api_key=os.environ["AZURE_OPENAI_API_VERSION"]
    )

from langchain_community.document_loaders import PyPDFLoader

# loading
async def loading(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages

from langchain.text_splitter import RecursiveCharacterTextSplitter

def splitting(pages):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
    all_splits = text_splitter.split_documents(pages)
    return all_splits

from langchain_openai import OpenAIEmbeddings  
from langchain_chroma import Chroma

def embedding(all_splits):
    embed=OpenAIEmbeddings(
    model="azure-text-embedding-3-large",  # or "text-embedding-ada-002"
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ['AZURE_OPENAI_ENDPOINT']
)   
    
    vectorstore_db = Chroma.from_documents(documents=all_splits, embedding=embed)
    return vectorstore_db

from langchain import hub
prompt = hub.pull("rlm/rag-prompt")

from typing import TypedDict, List
from langchain.schema import Document
# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State, vectorstore_db):
    retrieved_docs = vectorstore_db.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}



import asyncio
async def main():
    load = await loading("../database/avn-doc_database.pdf")
    splits = splitting(load)
    vector_db = embedding(splits)
    test_question = "Sử dụng bộ lọc để làm gì?"
    state = {"question": test_question, "context": [], "answer": ""}

    # Retrieve relevant docs
    result = retrieve(state, vector_db)
    state["context"] = result["context"]

    # Generate answer
    answer_result = generate(state)
    print("Answer:", answer_result["answer"])

if __name__ == "__main__":
    asyncio.run(main())