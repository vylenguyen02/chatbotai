from langchain_openai import OpenAIEmbeddings  
from langchain_community.vectorstores import Chroma
from service.chatbot.langchain.splitting_langchain import splitting
from service.chatbot.langchain.load_langchain import loading

async def embedding(file_name, embedder):
    load = await loading(f"../docs/{file_name}")
    all_splits = splitting(load)
    vectorstore_db = Chroma.from_documents(documents=all_splits, embedding=embedder, persist_directory="../database/chroma_db")
    vectorstore_db.persist()
    return vectorstore_db