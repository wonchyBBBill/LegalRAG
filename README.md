# Legal RAG System: Hong Kong Ordinances

This project implements a high-precision Retrieval-Augmented Generation (RAG) system designed to answer queries about Hong Kong legal ordinances. The system evolves from a naive implementation to a structural, high-performance pipeline capable of handling complex legal texts with high factual accuracy.

## 🚀 Project Evolution & Architecture

The system was developed in three stages to optimize the **Hit Rate** (retrieval) and **Pass Rate** (answer precision).

### Stage 1: Naive RAG (Baseline)
*   **Extraction**: Basic PDF text extraction.
*   **Chunking**: Fixed-size character splits (e.g., 500 chars).
*   **LLM**: Qwen 2.5-3B (via Ollama).
*   **Result**: Low hit rate due to split sentences; moderate pass rate.

### Stage 2: Structural RAG (Improved)
*   **Extraction**: PDF $\rightarrow$ Markdown conversion (via `pdf_to_markdown.py`) to preserve headers, lists, and tables.
*   **Chunking**: **Header-based splitting** (via `rebuild_index_md.py`). Instead of character counts, chunks are split by legal sections (`#`, `##`).
*   **LLM**: Qwen 2.5-3B (via Ollama).
*   **Result**: **100% Retrieval Hit Rate**. The system now consistently finds the correct legal section.

### Stage 3: High-Performance RAG (Optimized)
*   **LLM Upgrade**: Migrated from 3B to **Qwen 2.5-7B/8B** (via Hugging Face Transformers) to improve reasoning and extraction of precise numbers/terms.
*   **Prompt Engineering**: Implemented a **Hardened System Prompt** with strict refusal guidelines for out-of-scope questions and "Few-Shot" patterns for factual precision.
*   **Hardware**: Optimized for GPU acceleration using `float16` precision and `device_map="auto"`.
*   **Result**: Significant increase in **Pass Rate** and factual accuracy for complex legal queries.

---

## 📂 Repository Structure

### 🛠️ Core Pipeline
- `pdf_to_markdown.py`: Converts raw PDFs into structured Markdown files.
- `rebuild_index.py`: (Legacy) Build index from plain text.
- `rebuild_index_md.py`: **(Current)** Builds the vector index using structural Markdown headers.
- `vecdb.py`: Low-level vector database utility.

### 🤖 Chat Interfaces
- `RAGbot_ollama.py`: Lightweight chat interface using Ollama.
- `RAGbot_hf.py`: High-performance chat interface using Hugging Face Transformers.

### 🧪 Evaluation & Debugging
- `rag_eval_ollama.py`: Evaluation framework for the Ollama-based system.
- `rag_eval_hf.py`: Evaluation framework for the HF-based system (includes GPU support).
- `retrieval_debug.py`: Tool to inspect exactly which chunks are being retrieved for a specific query.
- `eval_dataset.json`: The gold-standard test set for benchmarking.

### 📂 Data
- `/law/`: Source PDF ordinances.
- `/law_md/`: Processed Markdown versions of the ordinances.
- `/law_db_v2/` or `/law_db_md/`: The Chroma vector databases.

---

## ⚙️ Setup & Usage

### 1. Installation
```bash
pip install langchain langchain-chroma langchain-community langchain-huggingface 
pip install transformers torch accelerate marker-pdf
```

### 2. The Pipeline Workflow
```bash
# Step 1: PDF -> Markdown
python pdf_to_markdown.py

# Step 2: Markdown -> Vector DB
python rebuild_index_md.py

# Step 3: Evaluate Performance
python rag_eval_hf.py --model "Qwen/Qwen2.5-7B-Instruct"

# Step 4: Chat with the Bot
python RAGbot_hf.py
```

## 📈 Key Findings
- **Structural Chunking > Fixed Chunking**: Splitting by headers improved retrieval hit rate from ~79% to 100%.
- **Model Size Matters**: Moving from 3B to 8B significantly reduced "summarization errors" and improved the extraction of specific legal terms and numbers.
- **Prompting is Critical**: Strict guidelines in the system prompt were required to prevent the model from hallucinating answers for out-of-scope questions.
