from service.chatbot.state_chatbot import State
from service.chatbot.embedding_chatbot import embedding
from service.chatbot.splitting_chatbot import splitting
from service.chatbot.load_chatbot import loading

import os

def retrieve(state: State, vectorstore_db):
    retrieved_docs = vectorstore_db.similarity_search(state["question"])
    return {"context": retrieved_docs}