import os
import sys
from langchain_openai import ChatOpenAI
sys.path.append('../..')

from langchain_community.vectorstores import Chroma

import service.chatbot.embedding_chatbot as embedding_chatbot, service.chatbot.retrieving_chatbot as retrieving_chatbot, service.chatbot.generating_chatbot as generating_chatbot

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

llm = ChatOpenAI(
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
        model=os.environ["AZURE_OPENAI_COMP_DEPLOYMENT_NAME"],
        api_key=os.environ["AZURE_OPENAI_API_VERSION"]
    )

def chroma_db_exists(path):
    return os.path.isdir(path) and len(os.listdir(path)) > 0

from langchain_openai import OpenAIEmbeddings  

import asyncio
async def main():
    file_name = input("Please enter file name: ")
    test_question = input("Ask something. ")
    persist_direct = "../database/chroma_db"

    embedder = OpenAIEmbeddings(
    model="azure-text-embedding-3-large",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ['AZURE_OPENAI_ENDPOINT']
)
    
    if os.path.exists(persist_direct) and os.listdir(persist_direct):
        vector_db = Chroma(persist_directory=persist_direct, embedding_function=embedder)
    else:
        vector_db = await embedding_chatbot.embedding(file_name, embedder)  

    state = {"question": test_question, "context": [], "answer": ""}

    # Retrieve relevant docs
    result = retrieving_chatbot.retrieve(state, vector_db)
    state["context"] = result["context"]

    # Generate answer
    answer_result = generating_chatbot.generate(state, llm)
    print(f'Answer: {result["context"]}')

if __name__ == "__main__":
    asyncio.run(main())