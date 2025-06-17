from langchain.text_splitter import RecursiveCharacterTextSplitter

def splitting(pages):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
    all_splits = text_splitter.split_documents(pages)
    return all_splits