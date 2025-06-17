from langchain_community.document_loaders import PyPDFLoader

# loading
async def loading(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages