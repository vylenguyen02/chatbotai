from langchain_openai import OpenAIEmbeddings  
from langchain_community.vectorstores import Chroma
from service.chatbot.langchain.splitting_langchain import splitting
from service.chatbot.langchain.load_langchain import loading

from langchain_mongodb import MongoDBAtlasVectorSearch


async def embedding(file_name, embedder,folder_path, collection, search_index):
    folder_path = folder_path = "../../docs"
    load = await loading(f"../docs/{file_name}")
    all_splits = splitting(load)         
    vectorstore_db = MongoDBAtlasVectorSearch.from_documents(
                    documents=all_splits,
                    embedding=embedder,
                    collection=collection,
                    index_name=search_index
                )
    ids=[val for val in range(len(all_splits))],
    vectorstore_db.add_documents(all_splits,ids=ids)
    return vectorstore_db