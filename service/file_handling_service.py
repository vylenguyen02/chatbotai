from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
import os
from typing import TypedDict, List
from langchain.schema import Document
from langchain import hub
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class FileHandlingService:
    def __init__(self, llm, pdf_path, embedder, collection, search_index):
        self.llm = llm
        self.pdf_path = os.path.abspath(pdf_path)
        self.embedder = embedder
        self.collecton = collection
        self.search_index = search_index

    def folder_has_any_file(self):
        for f in os.listdir(self.pdf_path):
            if os.path.isfile(os.path.join(self.pdf_path, f)) and not f.startswith("."):
                return True
        return False



    async def loading(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = []
        async for doc in loader.alazy_load():
            docs.append(doc)
        return docs
    
    def splitting(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
        all_splits = text_splitter.split_documents(docs)
        return all_splits

    def embed_documents(self, all_splits, embedder, collection, search_index):
        vectorstore = MongoDBAtlasVectorSearch.from_documents(
                    documents=all_splits,
                    embedding=embedder,
                    collection=collection,
                    index_name=search_index
                )
        ids=[str(val) for val in range(len(all_splits))],
        vectorstore.create_vector_search_index(dimensions=3072)
        vectorstore.add_documents(all_splits,ids=ids[0])
        return vectorstore

    def retrieve(self, state: State, vectorstore):
        retrieved_docs = vectorstore.similarity_search(state["question"])
        return {"context": retrieved_docs}
    
    def generate(self, state: State):
        prompt = hub.pull("rlm/rag-prompt")
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response.content}

# Define state for application
