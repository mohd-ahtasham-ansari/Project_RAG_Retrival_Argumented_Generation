from langchain_community.retrievers import ArxivRetriever

#create the retriever
retriever = ArxivRetriever(
    load_max_docs=2,#maximum number of paper to retrieve
    load_all_available_meta= True #to get all the avialable meta data
)
#query arxiv
docs = retriever.invoke("large language model")

#print result
for i, doc in enumerate(docs):
    print(f"\nresult {i+1}")
    print(f"title:{doc.metadata['Title']}")
    print(f"authors:{doc.metadata['Authors']}")
    print(f"Published:{doc.metadata['Published']}")
    print(f"summary:{doc.page_content}")
    print("="*50)
 