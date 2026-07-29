import os
from dotenv import load_dotenv,find_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(find_dotenv())

model = ChatMistralAI(model="mistral-small-2506")



template = ChatPromptTemplate(
    [("system"," You are an AI that summarizes the text"),
    ("human","{docs}")]
)


