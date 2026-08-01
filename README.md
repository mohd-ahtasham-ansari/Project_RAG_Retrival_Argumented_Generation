# Seneca RAG (Retrieval-Augmented Generation)

<!-- PASTE YOUR UI IMAGE BELOW THIS LINE -->

<br>
<br>

<!-- PASTE YOUR UI IMAGE ABOVE THIS LINE -->

This project is a sophisticated Retrieval-Augmented Generation (RAG) application that brings the timeless wisdom of Seneca, the Stoic philosopher, to life through an interactive chat interface. By integrating state-of-the-art language models with a curated knowledge base of Seneca's teachings (such as his Letters to Lucilius), the application can answer profound questions about life, ethics, and resilience. 

Unlike standard AI chatbots that rely solely on pre-training data, this RAG system dynamically retrieves highly relevant passages from actual source texts using ChromaDB. This retrieved context is then seamlessly fed into Mistral AI, which acts as the generation engine. A meticulously crafted system prompt ensures the AI adopts a stoic, patient, and deeply wise persona, offering advice and perspectives strictly grounded in Stoic philosophy. 

Whether you are seeking guidance on overcoming adversity, managing time, or finding inner peace, this application acts as a personal philosophical mentor, marrying ancient wisdom with modern artificial intelligence.

<img width="1175" height="746" alt="image" src="https://github.com/user-attachments/assets/b90bde6a-d5fc-4558-b8f7-79a5d0c0485d" />


## Features

- **Interactive UI**: A chat interface built with Streamlit (`app.py`).
- **Command-Line Interface**: A terminal-based chat script (`main.py`).
- **Document Processing**: Ingests PDF documents (like Seneca's Letters), splits them into chunks, and stores embeddings using ChromaDB (`create_database.py`).
- **Advanced Retrieval**: Utilizes Maximum Marginal Relevance (MMR) search to fetch diverse and relevant context from the vector database.
- **Mistral AI Integration**: Powered by Mistral AI for both embeddings and the language model generation.

## Requirements

- Python 3.8+
- Mistral AI API Key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mohd-ahtasham-ansari/Project_RAG_Retrival_Argumented_Generation.git
   cd Project_RAG_Retrival_Argumented_Generation
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables. Create a `.env` file in the project root and add your Mistral API key:
   ```env
   MISTRAL_API_KEY=your_api_key_here
   ```

## Usage

### 1. Create the Vector Database
Before running the application, you need to process your source documents (e.g., `Document loaders/book.pdf`) and create the Chroma vector database:
```bash
python create_database.py
```
This will generate a `Chroma_db` directory containing the stored embeddings.

### 2. Run the Streamlit Application (Web UI)
To launch the interactive chat interface:
```bash
streamlit run app.py
```

### 3. Run the CLI Application (Terminal)
To interact with the model directly via the terminal:
```bash
python main.py
```

## Architecture Overview
1. **Document Loading**: Uses `PyPDFLoader` to read the source PDFs.
2. **Text Splitting**: Uses `RecursiveCharacterTextSplitter` to create chunks for optimal embedding.
3. **Embeddings & Storage**: Uses `MistralAIEmbeddings` and `Chroma` to persist vector data.
4. **Retrieval & Generation**: Uses MMR search to pull context, which is then passed to `ChatMistralAI` along with a custom prompt to mimic Seneca's voice.
