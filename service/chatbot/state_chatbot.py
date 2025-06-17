from typing import TypedDict, List
from langchain.schema import Document
# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str