# Project Error Log

This document tracks significant errors faced during the development of the RAG (Retrieval-Augmented Generation) project and their resolutions.

## 2026-07-22: Package Installation Failure (`llvmlite`)

**Error:**
During `uv add -r requirements.txt`, the build failed for `llvmlite==0.36.0` with a build backend error. 

**Root Cause:**
`llvmlite` was being pulled in as a sub-dependency of `numba`. The specific version was failing to build in the current environment (likely due to Python version or missing system-level LLVM dependencies).

**Resolution:**
The conflicting package was removed/resolved, allowing `uv sync` and subsequent `uv add -r requirements.txt` to successfully install all other required packages (e.g., Langchain, FastAPI, PyTorch, Transformers).

---

*Note: Add any future environment setup, runtime, or dependency errors below this line to keep a historical record.*
