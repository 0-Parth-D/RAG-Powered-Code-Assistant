from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
import sys

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings,
        collection_name="rag_code_assistant"
    )
    return vectorstore

def load_prompt():
    template = (
        "You are an expert Python coding assistant. Use the following documentation "
        "excerpts to answer the user's question accurately. \n\n"
        "IMPORTANT INSTRUCTIONS:\n"
        "- Prioritize standard Python concepts over C-extensions or advanced typing unless specifically asked.\n"
        "- If multiple types of answers are in the context, synthesize them into a complete answer.\n"
        "- If the answer is not in the context, say 'I don't know'.\n\n"
        "Context:\n"
        "{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    return prompt

def load_llm():
    llm = Ollama(model="llama3", temperature=0.1)
    return llm

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def load_retriever(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Use MMR instead of standard similarity
        search_kwargs={
            "k": 4,         # Return 4 diverse chunks to the LLM
            "fetch_k": 20   # Search top 20, then pick the 4 most diverse
        }
    )
    return retriever

def load_rag_chain(retriever, prompt, llm):
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def load_answer(rag_chain, retriever, question):
    print("=== ASSISTANT'S ANSWER ===")
    
    # We use .stream() instead of .invoke() for the RAG chain
    # This yields words one by one as they are generated
    full_answer = ""
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)  # Print word immediately, without newlines
        full_answer += chunk
        
    print("\n") # Add a final newline when done
    
    # We still use .invoke() for the retriever to get the source documents
    # (Database retrieval happens instantly, no need to stream it)
    source_docs = retriever.invoke(question)
    
    return full_answer, source_docs

vectorstore = load_vectorstore()
prompt = load_prompt()
llm = load_llm()
retriever = load_retriever(vectorstore)
rag_chain = load_rag_chain(retriever, prompt, llm)

if __name__ == "__main__":
    vectorstore = load_vectorstore()
    prompt = load_prompt()
    llm = load_llm()
    retriever = load_retriever(vectorstore)
    rag_chain = load_rag_chain(retriever, prompt, llm)

    question = sys.argv[1] if len(sys.argv) > 1 else "What is Python?"
    print(f"\nUser Question: {question}\nThinking...\n")

    # load_answer now handles the streaming print
    answer, source_docs = load_answer(rag_chain, retriever, question)
    
    print("\n=== SOURCES USED ===")
    for idx, doc in enumerate(source_docs):
        source_file = doc.metadata.get("source", "Unknown file")
        print(f"{idx + 1}. {source_file}")
