# Project Error Log

This document tracks significant errors faced during the development of the RAG (Retrieval-Augmented Generation) project and their resolutions.

## 2026-07-22: Package Installation Failure (`llvmlite`)

**Error:**
During `uv add -r requirements.txt`, the build failed for `llvmlite==0.36.0` with a build backend error. 

**Root Cause:**
`llvmlite` was being pulled in as a sub-dependency of `numba`. The specific version was failing to build in the current environment (likely due to Python version or missing system-level LLVM dependencies).

**Resolution:**
The conflicting package was removed/resolved, allowing `uv sync` and subsequent `uv add -r requirements.txt` to successfully install all other required packages (e.g., Langchain, FastAPI, PyTorch, Transformers).

## 2026-07-27: Chroma Vector Store Initialization Errors

**Error:**
1. `NameError: name 'Document' is not defined`
2. `TypeError: Chroma.from_documents() missing 1 required positional argument: 'documents'`

**Root Cause:**
Missing import for LangChain's core `Document` object, and incorrect positional argument passing for `documents` in `Chroma.from_documents()`.

**Resolution:**
Added `from langchain_core.documents import Document` and passed the `documents` argument as a keyword argument (`documents=docs`) in `vector store/db.py`.

---

## 2026-07-27: ChatPromptTemplate Syntax Error

**Error:**
A syntax/type error occurred when initializing `ChatPromptTemplate` due to passing a set instead of a string template literal.

**Root Cause:**
The human message template was incorrectly formatted as `("human", {docs})` where `{docs}` created a Python set rather than a string literal. 

**Resolution:**
Converted the `docs` variable to a string template literal by enclosing it in quotes: `("human", "{docs}")`. 

## 2026-07-27: Prompt Template Variable Mapping Error

**Error:**
`KeyError` or `ValueError` related to missing input variables when formatting the prompt template.

**Root Cause:**
The `ChatPromptTemplate` was expecting a variable named `{docs}`, but the `format_messages` function was incorrectly called using `data=docs[0].page_content`.

**Resolution:**
Updated the parameter in `format_messages` to correctly map to the template variable: `docs=docs[0].page_content`.

## 2026-07-27: TextLoader Import Path Error

**Error:**
`ImportError: cannot import name 'TextLoader' from 'langchain_community'`

**Root Cause:**
`TextLoader` was being imported from the root `langchain_community` package, but it actually resides in the `document_loaders` submodule.

**Resolution:**
Corrected the import statement to `from langchain_community.document_loaders import TextLoader`.

---

*Note: Add any future environment setup, runtime, or dependency errors below this line to keep a historical record.*
