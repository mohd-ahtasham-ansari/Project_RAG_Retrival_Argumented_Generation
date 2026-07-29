# Retrieval-Augmented Generation (RAG) Roadmap

This roadmap covers the fundamental concepts, advanced techniques, and production considerations for building RAG applications.

## 1. Fundamentals
Before diving into RAG, it's essential to understand the core building blocks:
* **Large Language Models (LLMs):** Text generation, instruction tuning, prompt engineering.
* **Embeddings:** Dense vector representations of text. Models like OpenAI `text-embedding-ada-002`, Cohere, HuggingFace (BGE, MiniLM).
* **Vector Databases:** Systems optimized for storing and retrieving high-dimensional vectors (e.g., Pinecone, Chroma, Qdrant, Milvus, Weaviate, pgvector).
* **Frameworks:** LangChain, LlamaIndex, Haystack.

## 2. Naive RAG Architecture
The standard, baseline RAG pipeline consists of three main phases:

### A. Ingestion Pipeline (Data Prep)
- [x] **Document Loaders:** Extracting text from PDFs, HTML, Word docs, Markdown, Databases.
- [x] **Text Splitting / Chunking:** Breaking large documents into smaller, meaningful chunks (e.g., Character Splitter, Recursive Character Splitter, Semantic Splitter).
- [x] **Embedding:** Converting chunks into vector representations.
- [x] **Indexing:** Storing vectors and metadata in a Vector Database.

### B. Retrieval Pipeline
- [x] **Query Embedding:** Converting the user's query into the same vector space.
- [x] **Similarity Search:** Finding the top-k most similar chunks in the Vector DB (Cosine Similarity, Euclidean Distance, Dot Product).

### C. Generation Pipeline
1. **Prompt Augmentation:** Combining the retrieved context chunks with the user's original query.
2. **LLM Inference:** Passing the augmented prompt to the LLM to generate the final grounded response.

## 3. Advanced RAG
Naive RAG often suffers from low retrieval accuracy and hallucination. Advanced RAG techniques mitigate these issues.

### A. Pre-Retrieval (Query Optimization)
* **Query Expansion:** Expanding the query with synonyms or related concepts.
* **Query Transformation (Rewrite):** Using an LLM to rewrite a poorly phrased user query for better retrieval.
* **Multi-Query Retrieval:** Generating multiple variations of the query and retrieving documents for each.
* **Query Routing:** Routing the query to different datastores (e.g., Vector DB vs. SQL DB) based on intent.
* **Self-Querying:** Extracting metadata filters from the natural language query before hitting the vector DB.

### B. Retrieval Optimizations
* **Hybrid Search:** Combining dense (vector) search with sparse (keyword/BM25) search.
* **Parent-Child Chunking / Small-to-Big Retrieval:** Retrieving smaller, highly specific chunks but passing their larger parent document to the LLM for context.
* **Sentence Window Retrieval:** Fetching a matching sentence but including $N$ sentences before and after it for context.
* **Hierarchical Indices:** Routing queries through a summary index before querying the detailed chunk index.

### C. Post-Retrieval
* **Reranking:** Using a Cross-Encoder (e.g., Cohere Rerank, BGE Reranker) to score and reorder the retrieved chunks based on absolute relevance, filtering out noise.
* **Context Compression / Filtering:** Extracting only the relevant sentences from the retrieved chunks to save prompt space and reduce distraction.

## 4. Evaluation
You cannot improve what you cannot measure. RAG evaluation typically breaks down the pipeline into components.
* **Frameworks:** RAGAS, ARES, TruLens.
* **Key Metrics:**
    * **Context Relevance:** Did we retrieve the right information? (Precision/Recall).
    * **Faithfulness (No Hallucinations):** Is the generated answer fully supported by the retrieved context?
    * **Answer Relevance:** Does the generated answer directly address the user's query?

## 5. Agentic RAG
Moving beyond simple question answering to systems that can plan and use tools.
* **Tool Calling (Function Calling):** Letting the LLM decide when to use a retrieval tool vs. a calculator or an API.
* **Multi-Agent Systems:** Specialized agents handling different parts of the retrieval or synthesis process (e.g., AutoGen, LangGraph).

## 6. Production & MLOps
* **Caching:** Semantic caching (e.g., GPTCache) to serve identical or similar queries faster and cheaper.
* **Data Syncing:** Keeping the vector database updated as source documents change (Upserts/Deletes).
* **Security & Access Control:** Ensuring users only retrieve documents they are authorized to see.
* **Monitoring:** Tracking latency, costs, and feedback loops (thumbs up/down) in production.
