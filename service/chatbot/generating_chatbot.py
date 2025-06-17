from langchain import hub
prompt = hub.pull("rlm/rag-prompt")

from service.chatbot.state_chatbot import State
def generate(state: State, llm):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}