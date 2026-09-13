from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def main():
    # 1. Embeddings & Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # UPDATED: Now using law_db_md for better chunking (RecursiveCharacterTextSplitter)
    vector_db = Chroma(persist_directory="./law_db_md", embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5}) # Increased k slightly for better recall

    # 2. Local Ollama LLM
    llm = ChatOllama(model="qwen2.5:3b", temperature=0.1) # Lower temperature for higher factual consistency

    # 3. HARDENED System Prompt
    # We explicitly tell the model to refuse out-of-scope questions and stick strictly to the context.
    system_prompt = (
        "You are a strict Legal Assistant specializing in Hong Kong law. "
        "Your only source of truth is the provided context. "
        "Instructions:\n"
        "1. Use ONLY the provided context to answer the question.\n"
        "2. If the answer is not explicitly stated in the context, or if the question is about a topic "
        "(e.g., tax, personal injury, general advice) not present in the provided documents, "
        "you MUST state: 'I am sorry, but this information is outside the scope of the provided legal documents.'\n"
        "3. Do not summarize away critical details. Be precise.\n"
        "4. Do not use outside knowledge or hallucinations.\n"
        "5. If the context contains typos (e.g., 'Tade' instead of 'Trade'), interpret them based on the legal context.\n\n"
        "EXAMPLE OF A PERFECT ANSWER:\n"
        "Question: What is the term of protection for a patent?\n"
        "Context: ...The term of the patent is 20 years from the filing date...\n"
        "Answer: The term of protection for a patent is 20 years from the filing date.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 4. RAG Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("Legal RAG Bot Ready (Using law_db_md). How can I help you? (enter :q to quit)\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() == ":q":
            print("Bot: Bye!")
            break

        print("Thinking...", end="", flush=True)

        try:
            result = rag_chain.invoke({"input": question})
            answer = result["answer"]
            docs = result["context"]

            print("\r" + " " * 50 + "\r", end="") 
            print(f"Bot: {answer}\n")

            if docs:
                print("Refer to:")
                seen = set()
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", 0) + 1
                    key = (source, page)
                    if key not in seen:
                        seen.add(key)
                        print(f"  {i}. {source} page {page}")
                print()

        except Exception as e:
            print(f"\nError: {e}")
            print("Please ensure Ollama is running.\n")

if __name__ == "__main__":
    main()
