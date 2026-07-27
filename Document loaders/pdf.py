from langchain_community.document_loaders import PyPDFLoader

data= PyPDFLoader("Document loaders/book.pdf")

docs=data.load()
print(docs)
print(len(docs))