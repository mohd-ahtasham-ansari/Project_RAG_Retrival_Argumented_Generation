from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embedding_model = MistralAIEmbeddings(model="mistral-embed")

vectorstore = Chroma.from_documents(docs , embedding_model)

similarity_retriever = vectorstore.as_retriever(
    search_type= "similarity",
    search_kwargs={"k":2}    
)

print("\n =======similarity search result=======")
similarity_docs = similarity_retriever.invoke("what is used for data analysis")

for doc in similarity_docs:
    print(doc.page_content)

mmr_retriver = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":2 }
)

print("\n ======= mmr retriever result ======")

mmr_docs = mmr_retriver.invoke("what is used for data analysis")

for doc in mmr_docs:
    print(doc.page_content)