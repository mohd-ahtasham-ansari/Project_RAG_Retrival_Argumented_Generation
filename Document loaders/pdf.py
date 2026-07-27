from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data= PyPDFLoader("Document loaders/book.pdf")

docs=data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)

chunks = splitter.split_documents(docs)


print(chunks[10].page_content)