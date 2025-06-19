from service.chatbot.langchain.state_langchain import State


import os

def retrieve(state: State, vectorstore_db):
    retrieved_docs = vectorstore_db.similarity_search(state["question"])
    return {"context": retrieved_docs}