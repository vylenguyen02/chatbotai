from typing import Any

from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import os
path="../../docs/"
# Get elements
raw_pdf_elements = partition_pdf(
    filename=path + "LLaVA.pdf",
    # Using pdf format to find embedded image blocks
    extract_images_in_pdf=True,
    # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
    # Titles are any sub-section of the document
    infer_table_structure=True,
    # Post processing to aggregate text once we have the title
    chunking_strategy="by_title",
    # Chunking params to aggregate text blocks
    # Attempt to create a new chunk 3800 chars
    # Attempt to keep chunks > 2000 chars
    # Hard max on chunks
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=path,
)


class Element(BaseModel):
    type: str
    text: Any

import glob
import os

# Get all .txt file summaries


def main():
    category_counts = {}

    for element in raw_pdf_elements:
        category = str(type(element))
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1

    # Unique_categories will have unique elements
    unique_categories = set(category_counts.keys())

    categorized_elements = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            categorized_elements.append(Element(type="table", text=str(element)))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            categorized_elements.append(Element(type="text", text=str(element)))

    # Tables
    table_elements = [e for e in categorized_elements if e.type == "table"]

    # Text
    text_elements = [e for e in categorized_elements if e.type == "text"]

    prompt_text = """You are an assistant tasked with summarizing tables and text. \
    Give a concise summary of the table or text. Table or text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    model = ChatOpenAI(
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
        model=os.environ["AZURE_OPENAI_COMP_DEPLOYMENT_NAME"],
        api_key=os.environ["OPENAI_API_KEY"]
    )
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    # Apply to text
    texts = [i.text for i in text_elements]
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    # Apply to tables
    tables = [i.text for i in table_elements]
    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
    file_paths = glob.glob(os.path.expanduser(os.path.join(path, "*.txt")))

# Read each file and store its content in a list
    img_summaries = []
    for file_path in file_paths:
        with open(file_path, "r") as file:
            img_summaries.append(file.read())

    # Remove any logging prior to summary
    logging_header = "clip_model_load: total allocated memory: 201.27 MB\n\n"
    cleaned_img_summary = [s.split(logging_header, 1)[1].strip() if logging_header in s else s.strip()
    for s in img_summaries]

    embedder = OpenAIEmbeddings(
        model="azure-text-embedding-3-large",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ['AZURE_OPENAI_ENDPOINT']
    )

    vectorstore = Chroma(collection_name="summaries", embedding_function=embedder)

# The storage layer for the parent documents
    store = InMemoryStore()
    id_key = "doc_id"

    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Add texts
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(text_summaries)
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    summary_tables = []
    filtered_table_ids = []

    for i, s in enumerate(table_summaries):
        if s.strip():  # only keep if non-empty
            filtered_table_ids.append(str(uuid.uuid4()))
            summary_tables.append(
                Document(page_content=s, metadata={id_key: filtered_table_ids[-1]})
            )


    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(filtered_table_ids, tables)))

    # Add image summaries
    
    img_ids = [str(uuid.uuid4()) for _ in cleaned_img_summary]
    summary_img = [
        Document(page_content=s, metadata={id_key: img_ids[i]})
        for i, s in enumerate(cleaned_img_summary)
    ]
    retriever.vectorstore.add_documents(summary_img)
    ### Fetch images
    retriever.docstore.mset(list(zip(img_ids, cleaned_img_summary)))

    print(tables[2])
    retriever.invoke("Images / figures with playful and creative examples")[1]
if __name__ == "__main__":
    main()