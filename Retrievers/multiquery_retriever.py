from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings,ChatMistralAI
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

retriever = vectorstore.as_retriever()

model_llm = ChatMistralAI(model="mistral-small")

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm = model_llm,
)

query = "what is used for data analysis ? "

docs = multi_query_retriever.invoke(query)

print("\n ==== Result of multi_query_retriever ======")

for p in docs:
    print(p.page_content)
    print(p.metadata)