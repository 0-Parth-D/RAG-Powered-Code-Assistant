from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
import os

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings,
        collection_name="rag_code_assistant"
    )

def load_llm():
    # Make sure to use the dedicated ollama import to avoid JSON parsing errors
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(model="llama3.1", temperature=0.1, base_url=base_url)

def load_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20}
    )

def load_retriever_tool(retriever):
    # This built-in tool automatically accepts a "query" argument from the LLM,
    # searches the DB, and returns the raw text context back to the LLM.
    return create_retriever_tool(
        retriever, 
        "rag_retriever", 
        description="Retrieve relevant documents from the RAG database of programming languages documentations. Don't output raw JSON in your final answer."
    )

def load_agent(tools, llm):
    system_prompt = (
        "You are an expert all in one assistant. Follow these rules strictly:\n\n"
        "1. PYTHON QUESTIONS: YOU MUST use tools to search for the answer.\n"
        "2. GREETINGS: If the user says 'Hi' or 'Hello', respond warmly and ask how you can help with Python. DO NOT use the tool.\n"
        "3. OFF-TOPIC QUESTIONS: If the user asks a non-coding question (e.g., trivia, history), answer it briefly using your own knowledge, then politely steer the conversation back to Python. DO NOT use the tool.\n\n"
        "STRICT CONSTRAINTS:\n"
        "- NEVER output raw JSON in your final answer.\n"
        "- NEVER explain your internal workings or mention the terms 'tool', 'database', or 'training data' to the user.\n"
        "- NEVER apologize or say 'I am just an AI' or 'I don't have direct access'."
    )
    
    llm_with_tools = llm.bind_tools(tools)
    
    return create_agent(
        model=llm_with_tools,
        tools=tools,
        system_prompt=system_prompt,
    )

if __name__ == "__main__":
    # --- INITIALIZATION ---
    vectorstore = load_vectorstore()
    llm = load_llm()
    retriever = load_retriever(vectorstore)

    retriever_tool = load_retriever_tool(retriever)
    tools = [retriever_tool]

    agent = load_agent(tools, llm)

    # --- CONTINUOUS CHAT LOOP ---
    print("\n" + "="*50)
    print("🐍 Python Coding Assistant Initialized")
    print("Type 'quit', 'exit', or 'q' to end the conversation.")
    print("="*50 + "\n")

    chat_history = []

    while True:
        user_input = input("You: ")
        print("=== YOUR QUESTION ===")
        print(user_input)

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
            
        chat_history.append(HumanMessage(content=user_input))
        
        print("Thinking...\n")
        print("=== ASSISTANT'S ANSWER ===")
        
        try:
            full_response = ""
            for chunk, metadata in agent.stream(
                {"messages": chat_history},
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    print(chunk.content, end="", flush=True)
                    full_response += chunk.content

            print("\n" + "="*50 + "\n")
            
            chat_history.append(AIMessage(content=full_response))
            
        except Exception as e:
            print(f"\n[Error]: {e}")
            chat_history.pop()