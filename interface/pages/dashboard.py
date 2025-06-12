import streamlit as st
import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI
from openai import OpenAI
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

llm = ChatOpenAI(base_url=os.environ['BASE_URL'], model="azure-gpt-4o-mini")

name = st.text_input("Hi, what is your name?")

if name:
    st.write(f"Hi {name}, how can I help you?")

response = st.text_input("")
if response:
    response = llm.invoke(response)
    st.write(response.content)