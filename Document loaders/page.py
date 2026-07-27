from langchain_community.document_loaders import WebBaseLoader
 
url= "https://takeuforward.org/dsa/strivers-a2z-sheet-learn-dsa-a-to-z"

data = WebBaseLoader(url)
docs=data.load()

#print(len(docs))
print(docs[0].page_content)