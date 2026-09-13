"""
rebuild_index_md.py
===================
Builds the vector index using Markdown files and Header-based splitting.
This preserves the legal structure of the documents, leading to higher 
retrieval precision and better LLM synthesis.

Usage:
    python rebuild_index_md.py
"""

import os
import argparse
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_md_index(
    md_folder: str,
    db_path: str,
    embedding_model: str,
):
    # 1. Define the headers to split on
    # Legal documents typically use # for the title and ## or ### for sections/articles
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False # Keep headers in the text so the LLM knows which section it's reading
    )

    # We still use a recursive splitter as a backup for very long sections
    # to ensure we don't exceed the embedding model's token limit.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )

    all_docs = []
    md_files = [f for f in os.listdir(md_folder) if f.endswith(".md")]
    
    if not md_files:
        print(f"No markdown files found in {md_folder}")
        return

    for filename in md_files:
        path = os.path.join(md_folder, filename)
        print(f"Processing {filename}...")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # First split by headers (Structural split)
            header_splits = markdown_splitter.split_text(content)
            
            # Then split those sections if they are still too long (Size split)
            final_splits = text_splitter.split_documents(header_splits)
            
            # Add source metadata
            for doc in final_splits:
                doc.metadata["source"] = filename
                
            all_docs.extend(final_splits)
        except Exception as e:
            print(f"  ERROR processing {filename}: {e}")

    print(f"\nTotal structural chunks created: {len(all_docs)}")

    # 2. Embed & Store
    print(f"Embedding with '{embedding_model}'...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    if os.path.exists(db_path):
        print(f"WARNING: '{db_path}' already exists. Overwriting.")

    db = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=db_path,
    )
    db.persist()
    print(f"\nMarkdown index saved to '{db_path}'")
    print(f"Total vectors: {db._collection.count()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--md-folder", default="./law_md", help="Folder containing .md files")
    parser.add_argument("--out", default="./law_md_db", help="Output DB path")
    parser.add_argument("--embedding", default="all-MiniLM-L6-v2", help="Embedding model")
    args = parser.parse_args()

    build_md_index(
        md_folder=args.md_folder,
        db_path=args.out,
        embedding_model=args.embedding,
    )
    print(f"\nNext step: test this new MD index with:")
    print(f"  python rag_eval.py --db {args.out}")

if __name__ == "__main__":
    main()
