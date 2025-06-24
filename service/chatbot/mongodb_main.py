import os
import sys
from langchain_openai import ChatOpenAI
sys.path.append('../..')
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from service.chatbot.langchain.embedding_langchain import embedding
from service.chatbot.langchain.generating_langchain import generate
from service.chatbot.langchain.retrieving_langchain import retrieve

def folder_has_any_file(folder_path):
    return any(
        os.path.isfile(os.path.join(folder_path, f)) and not f.startswith(".")
        for f in os.listdir(folder_path)
    )

from langchain_openai import OpenAIEmbeddings  

import asyncio
async def main():
    client = MongoClient(os.environ["MONGODB_ATLAS_CLUSTER_URI"])

    db = os.environ["DB_NAME"]
    collection_name = os.environ["COLLECTION_NAME"]
    search_index = os.environ["ATLAS_VECTOR_SEARCH_INDEX_NAME"]

    collection = client[db][collection_name]


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
    vectorstore_db = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=os.environ["MONGODB_ATLAS_CLUSTER_URI"],
        namespace=f"{db}.{collection_name}",
        embedding=embedder,
        index_name=search_index,
    )
# Embed and upload to MongoDB Atlas
    
    folder_path = "../../docs"
    if folder_has_any_file("../../docs"):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                vectorstore_db = await embedding(file, embedder, collection, search_index)
                # os.remove(file_path)
    else:
        vectorstore_db = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embedder,  # same embedding model used when saving
            index_name=search_index,
            relevance_score_fn="cosine"
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