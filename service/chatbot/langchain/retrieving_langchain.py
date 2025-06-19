from service.chatbot.langchain.state_langchain import State
from service.chatbot.langchain.embedding_langchain import embedding
from service.chatbot.langchain.splitting_langchain import splitting
from service.chatbot.langchain.load_langchain import loading

import os

def retrieve(state: State, vectorstore_db):
    retrieved_docs = vectorstore_db.similarity_search(state["question"])
    return {"context": retrieved_docs}