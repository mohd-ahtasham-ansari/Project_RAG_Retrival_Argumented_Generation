import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")

import os
from dotenv import load_dotenv,find_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(find_dotenv())

docs = TextLoader("Document loaders/notes.txt").load()

template = ChatPromptTemplate(
    [("system"," You are an AI that summarizes the text"),
    ("human",{docs})]
)

model = ChatMistralAI(model="mistral-small-2506")

prompt =template.format_messages(data=docs[0].page_content)

result = model.invoke(prompt)

print(result.content)
