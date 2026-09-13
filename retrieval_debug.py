"""
Retrieval Debugger
==================
Inspect exactly which chunks the vector DB returns for any query —
without calling the LLM. Useful for diagnosing retrieval failures.

Usage:
    python retrieval_debug.py "What is the term of a patent?"
    python retrieval_debug.py "fair dealing" --k 8
    python retrieval_debug.py --compare "patent term" "copyright term"
"""

import argparse
import sys
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_retriever(db_path: str, k: int):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": k}), db


def show_results(query: str, retriever, db, verbose: bool = True):
    # Also get similarity scores via direct similarity_search_with_score
    docs_with_scores = db.similarity_search_with_score(query, k=retriever.search_kwargs["k"])

    print(f"\n{'='*60}")
    print(f"Query: {query!r}")
    print(f"Retrieved {len(docs_with_scores)} chunk(s):")
    print("=" * 60)

    for i, (doc, score) in enumerate(docs_with_scores, 1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", 0) + 1
        text   = doc.page_content.strip()

        # Chroma returns L2 distance (lower = more similar)
        print(f"\n[{i}] Source : {source}")
        print(f"     Page   : {page}")
        print(f"     Score  : {score:.4f}  (L2 distance — lower is better)")
        if verbose:
            print(f"     Content: {text[:400]}{'...' if len(text) > 400 else ''}")
        else:
            print(f"     Preview: {text[:100]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", nargs="*", help="Query strings to test")
    parser.add_argument("--db", default="./law_md_db")
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument("--compare", nargs=2, metavar=("Q1", "Q2"),
                        help="Compare retrieval for two queries side-by-side")
    parser.add_argument("--brief", action="store_true",
                        help="Show only source/page/score, not chunk text")
    args = parser.parse_args()

    retriever, db = load_retriever(args.db, args.k)

    if args.compare:
        for q in args.compare:
            show_results(q, retriever, db, verbose=not args.brief)
    elif args.queries:
        for q in args.queries:
            show_results(q, retriever, db, verbose=not args.brief)
    else:
        # Interactive mode
        print("Retrieval debugger — type a query, Ctrl+C to exit")
        while True:
            try:
                q = input("\nQuery> ").strip()
                if q:
                    show_results(q, retriever, db, verbose=not args.brief)
            except KeyboardInterrupt:
                print("\nBye.")
                break


if __name__ == "__main__":
    main()
