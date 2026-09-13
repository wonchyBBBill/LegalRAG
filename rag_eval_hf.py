"""
rag_eval_hf.py
===============================================
HIGH-PERFORMANCE VERSION of the RAG Evaluation Framework.
Replaces Ollama with a direct Hugging Face Transformers pipeline.
Designed to run on a high-performance server with a GPU.

Usage:
    python rag_eval_hf.py --model "Qwen/Qwen3-8B"
"""

import argparse
import json
import time
import csv
import torch
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── LangChain imports ─────────────────────────────────────────────────────────
from langchain_chroma import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    id: str
    question: str
    expected_keywords: list[str]
    source: str
    category: str
    expected_refusal: bool = False

@dataclass
class EvalResult:
    id: str
    question: str
    category: str
    answer: str = ""
    retrieved_sources: list[str] = field(default_factory=list)
    retrieved_pages: list[int] = field(default_factory=list)
    source_hit: Optional[bool] = None
    chunks_retrieved: int = 0
    keyword_score: float = 0.0
    keywords_found: list[str] = field(default_factory=list)
    keywords_missed: list[str] = field(default_factory=list)
    correctly_refused: Optional[bool] = None
    latency_s: float = 0.0
    passed: bool = False
    failure_reasons: list[str] = field(default_factory=list)

# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class RAGEvaluatorHF:
    REFUSAL_PHRASES = [
        "don't know", "do not know", "cannot answer", "can't answer",
        "not in my knowledge", "outside the scope", "no information",
        "not covered", "unable to find", "not provided"
    ]

    def __init__(
        self,
        db_path: str = "./law_md_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        hf_model_id: str = "Qwen/Qwen3-8B",
        k: int = 5,
    ):
        print("Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        print("Connecting to vector DB...")
        self.vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": k})

        print(f"Loading HF Model ({hf_model_id}) to GPU...")
        # 1. Load Tokenizer and Model
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            torch_dtype=torch.float16, # Use float16 for speed/memory on GPU
            device_map="auto",        # Automatically distribute across GPUs
            trust_remote_code=True
        )

        # 2. Create HF Pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # 3. Wrap in LangChain
        hf_llm = HuggingFacePipeline(pipeline=pipe)
        llm = ChatHuggingFace(llm=hf_llm)

        system_prompt = (
            "You are a strict Legal Assistant specializing in Hong Kong law. "
            "Your only source of truth is the provided context. "
            "Instructions:\n"
            "1. Use ONLY the provided context to answer the question.\n"
            "2. If the answer is not explicitly stated in the context, or if the question is about a topic "
            "(e.g., tax, personal injury, general advice) not present in the provided documents, "
            "you MUST state: 'I am sorry, but this information is outside the scope of the provided legal documents.'\n"
            "3. Do not use outside knowledge or hallucinations.\n"
            "4. If the context contains typos (e.g., 'Tade' instead of 'Trade'), interpret them based on the legal context.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        qa_chain = create_stuff_documents_chain(llm, prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, qa_chain)
        print("Ready.\n")

    def run_one(self, case: EvalCase) -> EvalResult:
        result = EvalResult(id=case.id, question=case.question, category=case.category)

        t0 = time.perf_counter()
        try:
            output = self.rag_chain.invoke({"input": case.question})
        except Exception as e:
            result.failure_reasons.append(f"Chain error: {e}")
            return result
        result.latency_s = round(time.perf_counter() - t0, 2)

        result.answer = output.get("answer", "")
        docs = output.get("context", [])
        result.chunks_retrieved = len(docs)

        seen = set()
        for doc in docs:
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", 0) + 1
            key = (src, page)
            if key not in seen:
                seen.add(key)
                result.retrieved_sources.append(src)
                result.retrieved_pages.append(page)

        if case.source not in ("None", "Multiple"):
            expected_src_fragment = case.source.lower().replace(" ", "")
            hit = any(expected_src_fragment in s.lower().replace(" ", "") for s in result.retrieved_sources)
            result.source_hit = hit
            if not hit:
                result.failure_reasons.append(f"Source miss: expected '{case.source}', got {result.retrieved_sources}")

        if case.expected_keywords:
            answer_lower = result.answer.lower()
            for kw in case.expected_keywords:
                if kw.lower() in answer_lower:
                    result.keywords_found.append(kw)
                else:
                    result.keywords_missed.append(kw)
            result.keyword_score = round(len(result.keywords_found) / len(case.expected_keywords), 2)
            if result.keyword_score < 0.5:
                result.failure_reasons.append(f"Low keyword score ({result.keyword_score}): missing {result.keywords_missed}")

        if case.expected_refusal:
            answer_lower = result.answer.lower()
            refused = any(phrase in answer_lower for phrase in self.REFUSAL_PHRASES)
            result.correctly_refused = refused
            if not refused:
                result.failure_reasons.append("Failed to refuse out-of-scope question")

        result.passed = len(result.failure_reasons) == 0
        return result

    def run_all(self, cases: list[EvalCase]) -> list[EvalResult]:
        results = []
        total = len(cases)
        for i, case in enumerate(cases, 1):
            print(f"[{i}/{total}] {case.id}: {case.question[:60]}...")
            r = self.run_one(case)
            status = "✓ PASS" if r.passed else "✗ FAIL"
            print(f"         {status}  |  latency: {r.latency_s}s  |  kw_score: {r.keyword_score}")
            if r.failure_reasons:
                for reason in r.failure_reasons:
                    print(f"           → {reason}")
            results.append(r)
        return results

def print_summary(results: list[EvalResult]):
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    avg_latency = round(sum(r.latency_s for r in results) / total, 2) if total else 0
    avg_kw = round(sum(r.keyword_score for r in results if r.keyword_score > 0) / max(1, sum(1 for r in results if r.keyword_score > 0)), 2)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (HF MODEL)")
    print("=" * 60)
    print(f"  Total tests     : {total}")
    print(f"  Passed          : {passed}  ({round(passed/total*100)}%)")
    print(f"  Failed          : {total - passed}")
    print(f"  Avg latency     : {avg_latency}s")
    print(f"  Avg kw score    : {avg_kw}")

    categories = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    print("\n  By category:")
    for cat, items in sorted(categories.items()):
        p = sum(1 for i in items if i.passed)
        print(f"    {cat:<20} {p}/{len(items)} passed")

    source_results = [r for r in results if r.source_hit is not None]
    if source_results:
        hits = sum(1 for r in source_results if r.source_hit)
        print(f"\n  Source retrieval hit rate: {hits}/{len(source_results)} ({round(hits/len(source_results)*100)}%)")

    refusal_results = [r for r in results if r.correctly_refused is not None]
    if refusal_results:
        correct = sum(1 for r in refusal_results if r.correctly_refused)
        print(f"  Out-of-scope refusal accuracy: {correct}/{len(refusal_results)}")

    failures = [r for r in results if not r.passed]
    if failures:
        print(f"\n  FAILED TESTS:")
        for r in failures:
            print(f"    [{r.id}] {r.question[:55]}...")
            for reason in r.failure_reasons:
                print(f"       - {reason}")
    print("=" * 60)

def load_cases(path: str = "eval_dataset.json") -> list[EvalCase]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [EvalCase(**item) for item in raw]

def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG legal Q&A system using HF Model")
    parser.add_argument("--db", default="./law_md_db", help="Path to Chroma vector DB")
    parser.add_argument("--dataset", default="eval_dataset.json", help="Eval dataset JSON")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="HF Model ID")
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    evaluator = RAGEvaluatorHF(db_path=args.db, hf_model_id=args.model)
    results = evaluator.run_all(cases)
    print_summary(results)

if __name__ == "__main__":
    main()
