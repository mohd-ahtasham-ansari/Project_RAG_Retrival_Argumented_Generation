from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size= 10,
    separator="",
    chunk_overlap=2
)

data = TextLoader("Document loaders/notes.txt")
#print(data)

docs = data.load()

chunks = splitter.split_documents(docs)

#print(docs[0].page_content)
print(len(chunks))

for i in chunks:
    print(i.page_content)
    print()