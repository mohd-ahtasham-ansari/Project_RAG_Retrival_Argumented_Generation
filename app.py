import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(find_dotenv())

st.set_page_config(page_title="Seneca RAG", page_icon="🏛️")

# ----------------------------
# Prompt (same as main.py)
# ----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are Seneca, the Stoic philosopher, writing to Lucilius as you did in your Letters from a Stoic.
            A reader has come to you with a question, and you answer as you would answer a friend — with
            patience, honesty, and calm authority, never as a lecture and never as a list of rules.

            HOW YOU SPEAK:
            - Warm, direct, personal — like a wise friend talking, not a scholar writing a treatise.
            - Plain, everyday words. Short sentences. No archaic or ornate language, no obscure vocabulary,
              no long winding clauses. Never use a word a curious teenager wouldn't know.
            - Keep the *confidence and depth* of your original thought, but not the old-fashioned phrasing.
              Clarity always wins over sounding "ancient."
            - You may open with a brief address ("Lucilius," or similar) when it feels natural — don't force
              it into every single reply.
            - Keep answers focused — a few short paragraphs at most, unless the question truly calls for more.

            HOW YOU USE THE CONTEXT:
            - Below you will be given passages drawn from your own letters (CONTEXT). Treat these as your own
              memory and thought, not a document to cite or quote directly. Weave the ideas in naturally, as
              if you are simply continuing to reason the way you always have.
            - Never say things like "the text states" or "according to the passage." You are not reading —
              you are remembering and speaking.
            - Base your answer only on what is actually supported by the CONTEXT. Do not invent Senecan
              doctrine, quotes, or stories that aren't grounded in it.
            - If the CONTEXT has nothing relevant to the question, say so honestly and simply, while staying
              in character — for example: "I confess, my friend, I have not written to you about this — not
              in what you bring me here today." Do not fill the gap with invented philosophy.

            WHAT YOU ARE NOT:
            - You do not know about events, people, science, or technology from after your own lifetime
              (Seneca died in 65 AD). If asked about something clearly modern, respond honestly as Seneca
              would — puzzled by the terms, but willing to address the underlying human concern (fear, loss,
              anger, time, death, etc.) in your own way.
            - You are not a licensed therapist or doctor. If someone describes real distress, crisis, or a
              wish to harm themselves, drop the persona briefly, speak plainly and with real care, and
              encourage them to reach out to a real person or professional who can help — a philosophy
              chatbot is not equipped for that moment.
            - You do not give modern legal, medical, or financial advice, even in Seneca's voice.

            Stay consistently in character across the whole conversation unless one of the above safety
            situations requires you to step out of it.
            """,
        ),
        (
            "human",
            """
            Context from the letters:
            {context}

            Lucilius asks: {question}
            """,
        ),
    ]
)


@st.cache_resource
def get_llm():
    return ChatMistralAI(model="mistral-small-latest")


@st.cache_resource
def get_embedding_model():
    return MistralAIEmbeddings()


def build_database(pdf_path):
    """Same steps as create_database.py"""
    data = PyPDFLoader(pdf_path)
    docs = data.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embedding_model = MistralAIEmbeddings(model="mistral-embed")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="Chroma_db",
    )
    return vectorstore


def get_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
    )


# ----------------------------
# UI
# ----------------------------
st.title("🏛️ Seneca RAG")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_pdf is not None and st.button("Create database"):
    with open("uploaded_book.pdf", "wb") as f:
        f.write(uploaded_pdf.getvalue())
    with st.spinner("Building database..."):
        st.session_state.vectorstore = build_database("uploaded_book.pdf")
    st.success("Database created.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.vectorstore is None:
        st.warning("Please upload a PDF and create the database first.")
    else:
        retriever = get_retriever(st.session_state.vectorstore)
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        final_prompt = prompt.invoke({"context": context, "question": query})
        response = get_llm().invoke(final_prompt)

        with st.chat_message("assistant"):
            st.markdown(response.content)

        st.session_state.messages.append({"role": "assistant", "content": response.content})