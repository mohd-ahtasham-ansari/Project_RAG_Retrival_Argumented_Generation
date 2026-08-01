import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(find_dotenv())

st.set_page_config(page_title="Seneca RAG", page_icon="🏛️")

embedding_model = MistralAIEmbeddings()

vectorstore = Chroma(
    persist_directory="Chroma_db",
    embedding_function=embedding_model,
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
)

llm = ChatMistralAI(model="mistral-small-latest")

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

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;0,700;1,500&family=EB+Garamond:ital@0;1&display=swap');

    :root {
        --white: #FFFFFF;
        --whitewash: #F7F9FA;
        --sea: #0E5A73;
        --sea-deep: #0A3D52;
        --sky: #4FA8C9;
        --gold: #C6A15B;
        --ink: #1C2B33;
    }

    .stApp {
        background: var(--whitewash);
        color: var(--ink);
    }

    /* Dome-arch header banner */
    .hero {
        position: relative;
        margin: -1rem -1rem 2.2rem -1rem;
        padding: 3.2rem 1rem 4rem 1rem;
        text-align: center;
        background: linear-gradient(180deg, var(--sea-deep) 0%, var(--sea) 55%, var(--sky) 100%);
        border-radius: 0 0 50% 50% / 0 0 60px 60px;
        box-shadow: 0 6px 24px rgba(10,61,82,0.18);
    }

    .hero h1 {
        font-family: 'Cormorant Garamond', serif !important;
        font-weight: 700 !important;
        color: var(--white) !important;
        font-size: 3.2rem !important;
        letter-spacing: 0.04em;
        margin: 0 !important;
        text-shadow: 0 2px 10px rgba(0,0,0,0.15);
    }

    .hero .eyebrow {
        font-family: 'EB Garamond', serif;
        font-style: italic;
        color: var(--gold);
        letter-spacing: 0.3em;
        text-transform: uppercase;
        font-size: 0.8rem;
        margin-bottom: 0.4rem;
    }

    .hero .subtitle {
        font-family: 'EB Garamond', serif;
        color: rgba(255,255,255,0.88);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    .hero .goldline {
        width: 90px;
        height: 2px;
        background: var(--gold);
        margin: 1rem auto 0 auto;
        opacity: 0.9;
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        background: var(--white);
        border: none;
        border-radius: 16px;
        box-shadow: 0 2px 10px rgba(14,90,115,0.08);
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.6rem;
    }

    [data-testid="stChatMessageContent"] p {
        font-family: 'EB Garamond', serif;
        font-size: 1.1rem;
        line-height: 1.6;
        color: var(--ink);
    }

    /* Distinguish user vs assistant bubbles */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #EAF5FA, #DCEEF5);
        border-left: 3px solid var(--sky);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        border-left: 3px solid var(--gold);
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        border: 1.5px solid var(--sky) !important;
        border-radius: 999px !important;
        background: var(--white) !important;
        box-shadow: 0 2px 8px rgba(14,90,115,0.10);
    }

    [data-testid="stChatInput"] textarea {
        font-family: 'EB Garamond', serif;
        font-size: 1.05rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Ἐπιστολαί</div>
        <h1>Letters from Seneca</h1>
        <div class="goldline"></div>
        <div class="subtitle">Ask, and receive the counsel of a Stoic</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("You:")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({"context": context, "question": query})
    response = llm.invoke(final_prompt)

    with st.chat_message("assistant"):
        st.markdown(response.content)

    st.session_state.messages.append({"role": "assistant", "content": response.content})