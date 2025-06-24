from service.chatbot.langchain.splitting_langchain import splitting
from service.chatbot.langchain.load_langchain import loading

from langchain_mongodb import MongoDBAtlasVectorSearch

import os 
async def embedding(file_name, embedder, collection, search_index):
    load = await loading(f"../../docs/{file_name}")
    all_splits = splitting(load)         
    vectorstore_db=MongoDBAtlasVectorSearch.from_documents(
                    documents=all_splits,
                    embedding=embedder,
                    collection=collection,
                    index_name=search_index
                )
    ids=[str(val) for val in range(len(all_splits))],
    vectorstore_db.add_documents(all_splits,ids=ids[0])
    return vectorstore_db