import asyncio
from service.file_handling_service import FileHandlingService
from langchain_openai import OpenAIEmbeddings  
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
import os
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
import uuid
from langchain_core.messages import SystemMessage
from langchain_mongodb import MongoDBChatMessageHistory



client = MongoClient(os.environ["MONGODB_ATLAS_CLUSTER_URI"])
db = os.environ["DB_NAME"]
collection_name = os.environ["COLLECTION_NAME"]
search_index = os.environ["ATLAS_VECTOR_SEARCH_INDEX_NAME"]
collection = client[db][collection_name]

embedder = OpenAIEmbeddings(
        model="azure-text-embedding-3-large",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ['AZURE_OPENAI_ENDPOINT']
    )

llm = ChatOpenAI(
            base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
            model=os.environ["AZURE_OPENAI_COMP_DEPLOYMENT_NAME"],
            api_key=os.environ["AZURE_OPENAI_API_VERSION"]
        )


async def main():
    self_service = FileHandlingService(llm, "docs/", embedder, collection, search_index)
    if self_service.folder_has_any_file():
            for file in os.listdir(self_service.pdf_path):
                new_file_path = os.path.join(self_service.pdf_path, file)
                if os.path.isfile(new_file_path):
                    docs = await self_service.loading(new_file_path)
                    all_splits = self_service.splitting(docs)
                    vectorstore_db = self_service.embed_documents(all_splits, embedder, collection, search_index)
                    os.remove(new_file_path)
    else:
         vectorstore_db = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embedder,  # same embedding model used when saving
            index_name=search_index,
            relevance_score_fn="cosine"
        )
        
    session_id = input("Enter your session ID: ").strip()
    if not session_id:
        session_id = str(uuid.uuid4())
        input_message = input("Ask something!\n")
        title_prompt = [SystemMessage(content="You're a helpful assistant. Based on the following message, generate a short and meaningful title (1–3 words) that describes the topic:\n\n" + input_message)
    ]
        generated_title = llm.invoke(title_prompt)
        generated_title = generated_title.content.strip().replace(" ", "_")
        session_id = f"{generated_title}_{str(uuid.uuid4())[:8]}"

    else: 
        input_message = input("Ask something!\n")
        
    history = MongoDBChatMessageHistory(
    connection_string=os.environ["MONGODB_ATLAS_CLUSTER_URI"],
    session_id=session_id
    )
    
    history.add_user_message(input_message)
    tools = ToolNode([self_service.make_retrieve(vectorstore_db)])

    graph = self_service.graph_building(tools, vectorstore_db)
    final_ai_message = None
    for step in graph.stream(
        {"messages": history.messages + [{"role": "user", "content": input_message}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
        if step["messages"][-1].type == "ai":
            final_ai_message = step["messages"][-1].content

    if final_ai_message:
        history.add_ai_message(final_ai_message)
        # print(final_ai_message)
if __name__ == "__main__":
    asyncio.run(main())
