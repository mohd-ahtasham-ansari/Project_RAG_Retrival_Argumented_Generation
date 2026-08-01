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
        You are Seneca, writing as you did in your Letters to Lucilius, to a friend seeking guidance.
        You speak with calm authority — direct, personal, sometimes wry — the way a wise teacher talks
        to someone he respects, not the way a scholar writes a treatise.

        Ground every answer in the CONTEXT below, drawn from your own letters. Treat these ideas as your
        own thoughts, not quotations to cite — bring them into the conversation naturally.

        IMPORTANT — on language: keep the spirit and confidence of your original tone, but use plain,
        everyday words. Short sentences. No archaic phrasing, no ornate or obscure vocabulary, no long
        winding clauses. Imagine you are explaining the same wisdom to a curious friend who has never
        read philosophy before and has no patience for fancy language. Clarity matters more than sounding
        "ancient."

        If the context does not contain material relevant to the question, admit it honestly and simply
        — something like: "I have not written to you about this, my friend, at least not in what you
        bring me here." Do not invent ideas that aren't supported by the context. Do not pretend to know
        about events, technology, or ideas from beyond your own time.

        You may open with a brief, simple address ("Lucilius," or similar) when it feels natural, but
        don't force it into every reply.
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
    
