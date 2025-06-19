import os
import sys
from langchain_openai import ChatOpenAI
sys.path.append('../..')

from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file



def folder_has_any_file(folder_path):
    return any(
        os.path.isfile(os.path.join(folder_path, f)) and not f.startswith(".")
        for f in os.listdir(folder_path)
    )


from langchain_community.document_loaders import PyPDFLoader

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

from langchain import hub
prompt = hub.pull("rlm/rag-prompt")

def generate(state: State, llm):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


from langchain_openai import OpenAIEmbeddings  

import asyncio
async def main():
    client = MongoClient(os.environ["MONGODB_ATLAS_CLUSTER_URI"])

    db = os.environ["DB_NAME"]
    collection = os.environ["COLLECTION_NAME"]
    search_index = os.environ["ATLAS_VECTOR_SEARCH_INDEX_NAME"]

    mongo_db = client[db][collection]


    llm = ChatOpenAI(
            base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
            model=os.environ["AZURE_OPENAI_COMP_DEPLOYMENT_NAME"],
            api_key=os.environ["AZURE_OPENAI_API_VERSION"]
        )

    test_question = input("Ask something. ")

    embedder = OpenAIEmbeddings(
        model="azure-text-embedding-3-large",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ['AZURE_OPENAI_ENDPOINT']
    )
    client = MongoClient(os.environ["MONGODB_ATLAS_CLUSTER_URI"])
    collection = client[os.environ["DB_NAME"]][os.environ["COLLECTION_NAME"]]

# Embed and upload to MongoDB Atlas
    
    folder_path = "../../docs"
    if folder_has_any_file("../../docs"):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                pages = await loading(file_path)
                all_splits = splitting(pages)
                vectorstore_db = MongoDBAtlasVectorSearch.from_documents(
                    documents=all_splits,
                    embedding=embedder,
                    collection=collection,
                    index_name=search_index
                )
                ids=[val for val in range(len(all_splits))],
                vectorstore_db.add_documents(all_splits,ids=ids)
                os.remove(file_path)
    else:
        vectorstore_db = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embedder,
            index_name=search_index,
            relevance_score_fn="cosine",
        )
    state = {"question": test_question, "context": [], "answer": ""}

    # Retrieve relevant docs
    result = retrieve(state, vectorstore_db)
    state["context"] = result["context"]

    # Generate answer
    answer_result = generate(state, llm)
    print(f'Answer: {result["context"]}')

if __name__ == "__main__":
    asyncio.run(main())