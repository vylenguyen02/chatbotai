from service.langchain.state_langchain import State

def make_retrieve(vector_store):
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}
    return retrieve