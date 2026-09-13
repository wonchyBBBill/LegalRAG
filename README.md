# Legal RAG System: Hong Kong Ordinances

This project implements a high-precision Retrieval-Augmented Generation (RAG) system designed to answer queries about Hong Kong legal ordinances. The system utilizes structural document parsing and a high-performance LLM to achieve maximum factual accuracy.

## 🚀 Project Architecture

The system follows a "Structural RAG" approach, moving beyond simple text splitting to preserve the inherent hierarchy of legal documents.

### The Pipeline
1.  **Structural Extraction**: PDFs are converted to Markdown (via `pdf_to_markdown.py`). This preserves headers, tables, and lists, which are critical for legal context.
2.  **Header-Based Indexing**: Instead of fixed-size chunks, the system uses `rebuild_index_md.py` to split documents by their actual legal sections (`#`, `##`, `###`). This ensures that legal rules are never split in half.
3.  **High-Precision Retrieval**: The system uses a Vector DB (Chroma) with `all-MiniLM-L6-v2` embeddings, achieving a **100% retrieval hit rate** on the evaluation dataset.
4.  **Controlled Generation**: A large-scale LLM (Qwen3-8B) is used with a **Hardened System Prompt**, enforcing strict adherence to the provided context and ensuring 100% accuracy in refusing out-of-scope questions.

---

## 📂 Repository Structure

### 🛠️ Core Pipeline
- `pdf_to_markdown.py`: Converts raw PDFs into structured Markdown files.
- `rebuild_index_md.py`: Builds the vector index using structural Markdown headers.

### 🤖 Chat Interfaces
- `RAGbot_ollama.py`: Lightweight chat interface using Ollama (best for local testing).
- `RAGbot_hf.py`: High-performance chat interface using Hugging Face Transformers (best for GPU servers).

### 🧪 Evaluation & Debugging
- `rag_eval_ollama.py`: Benchmarks the system using the Ollama backend.
- `rag_eval_hf.py`: Benchmarks the system using the Hugging Face backend.
- `retrieval_debug.py`: Tool to inspect exactly which chunks are being retrieved for a specific query.
- `eval_dataset.json`: The gold-standard test set used to measure hit rate and pass rate.

### 📂 Data
- `/law/`: Source PDF ordinances.
- `/law_md/`: Processed Markdown versions of the ordinances.
- `/law_md_db/`: The structural Chroma vector database.

---

## ⚙️ Setup & Usage

### 1. Installation
```bash
pip install langchain langchain-chroma langchain-community langchain-huggingface langchain-classic
pip install transformers torch accelerate marker-pdf
```

### 2. Execution Workflow
```bash
# Step 1: PDF -> Markdown
python pdf_to_markdown.py

# Step 2: Markdown -> Vector DB
python rebuild_index_md.py

# Step 3: Evaluate Performance
python rag_eval_hf.py

# Step 4: Chat with the Bot
python RAGbot_hf.py
```

## 📈 Performance Benchmarks
- **Retrieval Hit Rate**: 100% (Improved from 79% via structural chunking).
- **Out-of-Scope Refusal**: 100% (Achieved via system prompt hardening).
- **Pass Rate**: Significantly increased by migrating from 3B $\rightarrow$ 8B parameter models.
