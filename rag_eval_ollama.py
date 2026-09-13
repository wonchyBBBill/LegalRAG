"""
RAG Evaluation Framework for Legal Q&A System
===============================================
Tests retrieval quality, answer relevance, keyword coverage,
out-of-scope refusal, and latency across all eval categories.

Usage:
    python rag_eval.py                   # full eval run
    python rag_eval.py --category factual  # filter by category
    python rag_eval.py --id pat-001       # run a single test
    python rag_eval.py --report           # save results to CSV
"""

import argparse
import json
import time
import csv
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── LangChain imports (same stack as RAGbot.py) ─────────────────────────────
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

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

    # Retrieval metrics
    source_hit: Optional[bool] = None      # Did we retrieve from the right document?
    chunks_retrieved: int = 0

    # Answer metrics
    keyword_score: float = 0.0             # Fraction of expected keywords found
    keywords_found: list[str] = field(default_factory=list)
    keywords_missed: list[str] = field(default_factory=list)
    correctly_refused: Optional[bool] = None  # For out-of-scope questions

    # Latency
    latency_s: float = 0.0

    # Overall pass/fail
    passed: bool = False
    failure_reasons: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class RAGEvaluator:
    REFUSAL_PHRASES = [
        "don't know", "do not know", "cannot answer", "can't answer",
        "not in my knowledge", "outside the scope", "no information",
        "not covered", "unable to find", "not provided"
    ]

    def __init__(
        self,
        db_path: str = "./law_db_md", # UPDATED: default to v2
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "qwen2.5:3b",
        k: int = 5, # UPDATED: match bot's k=5
    ):
        print("Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        print("Connecting to vector DB...")
        self.vector_db = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings,
        )
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": k})

        print(f"Loading LLM ({llm_model})...")
        llm = ChatOllama(model=llm_model, temperature=0.1) # UPDATED: match bot's temp=0.1

        # UPDATED: Hardened system prompt to match RAGbot2.py
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

    # ── Core evaluation logic ─────────────────────────────────────────────────

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

        # ── Collect retrieved sources ─────────────────────────────────────
        seen = set()
        for doc in docs:
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", 0) + 1
            key = (src, page)
            if key not in seen:
                seen.add(key)
                result.retrieved_sources.append(src)
                result.retrieved_pages.append(page)

        # ── Source hit (skip for cross-doc and out-of-scope) ─────────────
        if case.source not in ("None", "Multiple"):
            expected_src_fragment = case.source.lower().replace(" ", "")
            hit = any(
                expected_src_fragment in s.lower().replace(" ", "")
                for s in result.retrieved_sources
            )
            result.source_hit = hit
            if not hit:
                result.failure_reasons.append(
                    f"Source miss: expected '{case.source}', got {result.retrieved_sources}"
                )

        # ── Keyword coverage ──────────────────────────────────────────────
        if case.expected_keywords:
            answer_lower = result.answer.lower()
            for kw in case.expected_keywords:
                if kw.lower() in answer_lower:
                    result.keywords_found.append(kw)
                else:
                    result.keywords_missed.append(kw)
            result.keyword_score = round(
                len(result.keywords_found) / len(case.expected_keywords), 2
            )
            if result.keyword_score < 0.5:
                result.failure_reasons.append(
                    f"Low keyword score ({result.keyword_score}): "
                    f"missing {result.keywords_missed}"
                )

        # ── Out-of-scope refusal ──────────────────────────────────────────
        if case.expected_refusal:
            answer_lower = result.answer.lower()
            refused = any(phrase in answer_lower for phrase in self.REFUSAL_PHRASES)
            result.correctly_refused = refused
            if not refused:
                result.failure_reasons.append(
                    "Failed to refuse out-of-scope question"
                )

        # ── Final pass/fail ───────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list[EvalResult]):
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    avg_latency = round(sum(r.latency_s for r in results) / total, 2) if total else 0
    avg_kw = round(
        sum(r.keyword_score for r in results if r.keyword_score > 0) /
        max(1, sum(1 for r in results if r.keyword_score > 0)), 2
    )

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total tests     : {total}")
    print(f"  Passed          : {passed}  ({round(passed/total*100)}%)")
    print(f"  Failed          : {total - passed}")
    print(f"  Avg latency     : {avg_latency}s")
    print(f"  Avg kw score    : {avg_kw}")

    # Per-category breakdown
    categories = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    print("\n  By category:")
    for cat, items in sorted(categories.items()):
        p = sum(1 for i in items if i.passed)
        print(f"    {cat:<20} {p}/{len(items)} passed")

    # Source hit rate
    source_results = [r for r in results if r.source_hit is not None]
    if source_results:
        hits = sum(1 for r in source_results if r.source_hit)
        print(f"\n  Source retrieval hit rate: {hits}/{len(source_results)} "
              f"({round(hits/len(source_results)*100)}%)")

    # Refusal accuracy
    refusal_results = [r for r in results if r.correctly_refused is not None]
    if refusal_results:
        correct = sum(1 for r in refusal_results if r.correctly_refused)
        print(f"  Out-of-scope refusal accuracy: {correct}/{len(refusal_results)}")

    # Failures detail
    failures = [r for r in results if not r.passed]
    if failures:
        print(f"\n  FAILED TESTS:")
        for r in failures:
            print(f"    [{r.id}] {r.question[:55]}...")
            for reason in r.failure_reasons:
                print(f"       - {reason}")

    print("=" * 60)


def save_csv(results: list[EvalResult], path: str = "eval_results.csv"):
    if not results:
        return
    rows = []
    for r in results:
        rows.append({
            "id": r.id,
            "category": r.category,
            "passed": r.passed,
            "keyword_score": r.keyword_score,
            "source_hit": r.source_hit,
            "correctly_refused": r.correctly_refused,
            "latency_s": r.latency_s,
            "chunks_retrieved": r.chunks_retrieved,
            "keywords_found": ", ".join(r.keywords_found),
            "keywords_missed": ", ".join(r.keywords_missed),
            "failure_reasons": "; ".join(r.failure_reasons),
            "answer_snippet": r.answer[:200],
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def load_cases(path: str = "eval_dataset.json") -> list[EvalCase]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [EvalCase(**item) for item in raw]


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG legal Q&A system")
    parser.add_argument("--db", default="./law_db_md", help="Path to Chroma vector DB")
    parser.add_argument("--dataset", default="eval_dataset.json", help="Eval dataset JSON")
    parser.add_argument("--category", help="Filter by category (factual, procedural, etc.)")
    parser.add_argument("--id", help="Run a single test by ID")
    parser.add_argument("--report", action="store_true", help="Save CSV report")
    args = parser.parse_args()

    cases = load_cases(args.dataset)

    if args.id:
        cases = [c for c in cases if c.id == args.id]
        if not cases:
            print(f"No test with id '{args.id}'")
            return
    elif args.category:
        cases = [c for c in cases if c.category == args.category]

    if not cases:
        print("No matching test cases.")
        return

    evaluator = RAGEvaluator(db_path=args.db)
    results = evaluator.run_all(cases)
    print_summary(results)

    if args.report:
        save_csv(results)


if __name__ == "__main__":
    main()
