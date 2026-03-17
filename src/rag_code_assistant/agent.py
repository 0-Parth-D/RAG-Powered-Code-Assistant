import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from fastapi.security import APIKeyHeader
from fastapi import FastAPI, UploadFile, File, Security, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_pinecone import PineconeVectorStore # Changed from Chroma
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import fast_tokenizer
from pathlib import Path
from langchain_community.document_loaders import UnstructuredHTMLLoader, UnstructuredMarkdownLoader, TextLoader, BSHTMLLoader

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return PineconeVectorStore(
        index_name="rag-agent",
        embedding=embeddings,
        pinecone_api_key=os.environ["PINECONE_API_KEY"],
    )

def load_llm():
    """
    Loads the LLM with fallback logic:
    - Tries Ollama (local development with your laptop)
    - Falls back to Groq Cloud (production deployment on Hugging Face)
    """
    ollama_url = os.environ["OLLAMA_BASE_URL"]
    
    # If OLLAMA_BASE_URL is set, use local Ollama (for demo purposes)
    if ollama_url:
        print("🔧 Using local Ollama LLM (Development Mode)")
        return ChatOllama(
            model="llama3.1",
            temperature=0.1,
            base_url=ollama_url,
        )
    
    # Otherwise, use Groq Cloud (for production on Hugging Face)
    groq_api_key = os.environ["GROQ_API_KEY"]
    if not groq_api_key:
        raise ValueError(
            "Neither OLLAMA_BASE_URL nor GROQ_API_KEY found! "
            "Please set one in your environment variables."
        )
    
    print("☁️  Using Groq Cloud LLM (Production Mode)")
    return ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",  # Fast, smart, and free!
        temperature=0.1
    )

def load_retriever(vectorstore):
    # Kept exactly as you wrote it
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20}
    )

def load_retriever_tool(retriever):
    # Kept exactly as you wrote it
    return create_retriever_tool(
        retriever, 
        "rag_retriever", 
        description="Retrieve relevant documents from the RAG database of programming languages documentations. Don't output raw JSON in your final answer."
    )

def load_agent(tools, llm):
    # Kept exactly as you wrote it
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


# --- FASTAPI SETUP & GLOBAL INITIALIZATION ---

app = FastAPI(title="Python RAG Agent API")

# 1. Define the name of the header we expect
api_key_header = APIKeyHeader(name="X-API-Key")

# 2. Get your secret password from environment variables
SECRET_APP_KEY = os.environ["APP_API_KEY"]

# 3. Create the security function
def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != SECRET_APP_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize your agent once when the server starts
vectorstore = load_vectorstore()
llm = load_llm()
retriever = load_retriever(vectorstore)
retriever_tool = load_retriever_tool(retriever)
tools = [retriever_tool]
agent = load_agent(tools, llm)


# --- API ENDPOINTS ---

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # Allows UI to send previous messages

@app.post("/chat", dependencies=[Depends(verify_api_key)])
async def chat_endpoint(request: ChatRequest):
    # 1. Build the chat history array from the UI's request
    chat_history = []
    for msg in request.history:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))
            
    chat_history.append(HumanMessage(content=request.message))

    # 2. Wrap your exact original streaming logic in a generator function
    async def generate_stream():
        try:
            for chunk, metadata in agent.stream(
                {"messages": chat_history},
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    yield chunk.content
        except Exception as e:
            yield f"\n[Error]: {e}"

    # 3. Stream the output to the Vercel frontend
    return StreamingResponse(generate_stream(), media_type="text/event-stream")


# 1. Add your custom token length function back
def custom_token_length(text):
    tokens = fast_tokenizer.tokenize(text)
    return len(tokens)

@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_document(file: UploadFile = File(...)):
    """Accepts PDF, HTML, MD, and TXT files and uploads them to Pinecone using fast_tokenizer."""
    
    ext = Path(file.filename).suffix.lower()
    
    supported_extensions = [".pdf", ".html", ".htm", ".md", ".txt"]
    if ext not in supported_extensions:
        return {"error": f"Unsupported file type. Please upload one of: {', '.join(supported_extensions)}"}

    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            
        elif ext in [".html", ".htm"]:
            try:
                loader = UnstructuredHTMLLoader(temp_file_path)
                docs = loader.load()
            except Exception as e:
                print(f"Warning: UnstructuredHTMLLoader failed, trying BSHTMLLoader: {e}")
                loader = BSHTMLLoader(temp_file_path)
                docs = loader.load()
                
        elif ext == ".md":
            loader = UnstructuredMarkdownLoader(temp_file_path)
            docs = loader.load()
            
        elif ext == ".txt":
            loader = TextLoader(temp_file_path)
            docs = loader.load()

        # 2. Re-implement your exact RecursiveCharacterTextSplitter settings
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,       # Max 350 tokens per chunk
            chunk_overlap=50,     # Overlap of 50 tokens
            length_function=custom_token_length  # Tells LangChain to use your C++ tool
        )
        
        splits = text_splitter.split_documents(docs)

        # 3. Upload the perfectly tokenized chunks to Pinecone
        vectorstore.add_documents(splits)

        return {
            "status": "success", 
            "message": f"Successfully processed {file.filename} into {len(splits)} chunks and uploaded to Pinecone."
        }
        
    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


if __name__ == "__main__":
    # Runs the API server on port 7860 (Required for Hugging Face Spaces)
    print("\n" + "="*50)
    print("🐍 Python RAG API Initialized on Port 7860")
    print("="*50 + "\n")
    uvicorn.run("agent:app", host="0.0.0.0", port=7860, reload=True)