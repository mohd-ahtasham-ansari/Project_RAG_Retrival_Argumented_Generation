# Project RAG (Retrieval-Augmented Generation)

This project demonstrates a simple document processing pipeline using LangChain, Mistral AI, and Python. It sets the foundation for a Retrieval-Augmented Generation (RAG) system.

## Overview

The application is designed to:
1. Load documents (PDF files) using `PyPDFLoader`.
2. Split the document text into manageable chunks using `RecursiveCharacterTextSplitter`.
3. Generate embeddings using `MistralAIEmbeddings` and store/retrieve them via `Chroma` vector store.
4. Use LangChain along with a Mistral AI model (`ChatMistralAI`) to process the text.
5. Summarize or extract information from the parsed documents based on a provided prompt template.

## Requirements

- Python 3.8+
- Mistral AI API Key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mohd-ahtasham-ansari/Project_RAG_Retrival_Argumented_Generation.git
   cd Project_RAG_Retrival_Argumented_Generation
   ```
2. Create and activate a virtual environment (optional but recommended).
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your environment variables. Create a `.env` file in the project root and add your Mistral API key:
   ```env
   MISTRAL_API_KEY=your_api_key_here
   ```

## Usage

Ensure you have a PDF file available (e.g., `Document loaders/book.pdf`). Then run the main script to process the document and generate a summary:

```bash
python main.py
```

To test the vector store functionality (embedding, storing, and retrieving):
```bash
python "vector store/db.py"
```
