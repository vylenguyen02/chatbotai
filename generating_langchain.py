from langchain import hub
prompt = hub.pull("rlm/rag-prompt")

from service.langchain.state_langchain import State
def make_generate(llm):
    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}
    return generate