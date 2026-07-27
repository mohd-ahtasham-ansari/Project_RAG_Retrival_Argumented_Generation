import os
from dotenv import load_dotenv,find_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(find_dotenv())

model = ChatMistralAI(model="mistral-small-2506")

docs = PyPDFLoader("Document loaders/book.pdf").load()

template = ChatPromptTemplate(
    [("system"," You are an AI that summarizes the text"),
    ("human","{docs}")]
)

prompt =template.format_messages(docs=docs[15].page_content)

result = model.invoke(prompt)

print(result.content)
