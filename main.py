import asyncio
from service.file_handling_service import FileHandlingService
from langchain_openai import OpenAIEmbeddings  
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
import os


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

if __name__ == "__main__":
    asyncio.run(main())
