import os
import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv())

# Initialize LLM
llm = ChatOpenAI(base_url=os.environ["BASE_URL"], model="azure-gpt-4o-mini")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input prompt
prompt = st.chat_input("Ask a question")
if prompt:
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get assistant response
    response = llm.invoke(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        st.write(response.content)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": response.content})
