import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")

import os
from dotenv import load_dotenv,find_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader

load_dotenv(find_dotenv())

docs = TextLoader("Document loaders/notes.txt").load()

model = ChatMistralAI(model="mistral-small-2506")

result = model.invoke("hello")

print(docs[0].page_content)

print(result.content)
