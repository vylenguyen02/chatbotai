import os
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
from langchain_community.vectorstores import Chroma
def embedding(all_splits, embedder):
    vectorstore_db = Chroma.from_documents(documents=all_splits, embedding=embedder, persist_directory="../database/chroma_db")
    vectorstore_db.persist()
    return vectorstore_db

def chroma_db_exists(path):
    return os.path.isdir(path) and len(os.listdir(path)) > 0

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
    file_name = input("Please enter file name: ")
    test_question = input("Ask something. ")
    file_path = "../database/" + file_name
    persist_direct = "../database/chroma_db"

    embedder = OpenAIEmbeddings(
    model="azure-text-embedding-3-large",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ['AZURE_OPENAI_ENDPOINT']
)
    
    if os.path.exists(persist_direct) and os.listdir(persist_direct):
        vector_db = Chroma(persist_directory=persist_direct, embedding_function=embedder)
    else:
        load = await loading(f"../database/{file_name}")
        splits = splitting(load)
        vector_db = embedding(splits, embedder)  
    
    state = {"question": test_question, "context": [], "answer": ""}

    # Retrieve relevant docs
    result = retrieve(state, vector_db)
    state["context"] = result["context"]

    # Generate answer
    answer_result = generate(state)
    print(f'Context: {state["context"]}\n\n')
    print(f'Answer: {result["context"]}')

if __name__ == "__main__":
    asyncio.run(main())