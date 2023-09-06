import streamlit as st
import pandas as pd
import plotly.express as px

prompt = st.chat_input("Say something")
if prompt:
    st.write(f"User has sent the following prompt: {prompt}")
    print(prompt)
    prompt = str(prompt)

st.write('# QA langchain')
st.markdown('''
    This is a dashboard showing the *Q&A* over Documents using langchain :rocket:  
    code source: [Colab](https://colab.research.google.com/drive/1Eh6XRoE80MMRKeaJiugXNljpEdqws4tL#scrollTo=9pSPlYwvnQaP)
    ''')

from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://cogitotech.medium.com/how-do-self-driving-cars-work-abcac21ececb")
data = loader.load()
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 100,
    chunk_overlap  = 20,
    length_function = len,
    add_start_index = True,
)
texts = text_splitter.split_documents(data)
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import os
embeddings = OpenAIEmbeddings(openai_api_key ="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") # set openai_api_key = 'your_openai_api_key'
pinecone.init(api_key="5141cd92-aadf-4521-8819-ef21310d2722", environment="gcp-starter")
index_name = pinecone.Index('index-1')
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name='gpt-3.5-turbo-0301', temperature=0,openai_api_key = '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')
prediction = llm.predict(str(prompt))
print (prediction)

with st.chat_message("user"):
    st.write(str(prediction))



