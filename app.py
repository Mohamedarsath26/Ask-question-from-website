import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import bs4
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

## setup groq api key
groq_api_key = os.getenv('GROQ_API_KEY')

## select the model
llm = ChatGroq(model_name='llama-3.1-70b-versatile',groq_api_key=groq_api_key)
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

## select the embedding
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

def ask_question_with_url(url,question):
        ### load the website url
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(),
        )

        ## load the document
        docs=loader.load()

        ## split the document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        ## convert document to vector and store it to chromadb

        vector_store = Chroma.from_documents(embedding=embeddings,documents=splits)

        ## create the retriver
        retriver = vector_store.as_retriever()

        ## prompt
        system_prompt = (
            "you are AI assistant answer the question based on context"
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
        [
            ('system',system_prompt),
            ('human',"{input}")
        ]
        )

        ## creating stufff document chain
        question_answer_chain = create_stuff_documents_chain(llm,prompt)

        ## create retrival chain
        rag_chain = create_retrieval_chain(retriver,question_answer_chain)

        ## response
        response = rag_chain.invoke({'input':question})

        return response['answer']



## streamlit

st.title("Ask Question From Any Website")

url = st.text_input("Enter the Url")

question = st.text_input("Ask a Question")

if url and question:
     if st.button('**Get The Answer**'):   
        response = ask_question_with_url(url,question)
        st.success(response)

elif url:
      st.warning("Please ask question")

else:
      st.warning("Please provide Url and Question")
    



