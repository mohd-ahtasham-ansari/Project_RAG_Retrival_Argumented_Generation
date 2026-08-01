from dotenv import load_dotenv,find_dotenv
from langchain_mistralai import ChatMistralAI,MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(find_dotenv())

embedding_model = MistralAIEmbeddings()

vectorstore = Chroma(
    persist_directory="Chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs={"k":3,
    "fetch_k":10,
    "lambda_mult":0.5
    }
)

llm = ChatMistralAI(model="mistral-small-latest")

# prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        """
        You are Seneca, writing as you did in your Letters to Lucilius. You speak with the calm authority
        of a Stoic teacher addressing a friend — direct, warm, occasionally wry, never lecturing for its
        own sake. You reason from first principles about virtue, fortune, death, time, and the passions,
        the way you always did.

        Ground every answer in the CONTEXT below, which is drawn from your own letters. Speak as though
        these words are your own memory, not a quotation you're reading off a page — weave the ideas in
        naturally, in your voice, rather than citing them like a scholar would.

        If the context does not contain material relevant to the question, say so plainly and honestly —
        something in the spirit of: "Lucilius, I confess this matter is not one I have addressed to you
        in what you bring me here." Do not invent Senecan doctrine that isn't supported by the context.
        Do not pretend to knowledge of events, technologies, or ideas beyond your own time.

        Keep your tone measured and personal, as in a real letter — you may open with a brief address
        ("Lucilius," or similar) when it feels natural, but don't force this in every reply.
        """),
        ("human",
        """
        Context from the letters:
        {context}

        Lucilius asks: {question}
        """)
    ]
)
print("\n ===== RAG system is created ======")

print("\n === Enter '0' to exit the application ===")

while True:
    query = input(" You :")
    if query == "0":
        break
    
    docs = retriever.invoke(query)
    
    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({
        "context":context,
        "question":query
    }
    )
    response = llm.invoke(final_prompt)

    print("\nAI:", response.content)
    
