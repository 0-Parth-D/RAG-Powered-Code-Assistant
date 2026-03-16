import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import UnstructuredHTMLLoader, UnstructuredMarkdownLoader, TextLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pathlib import Path
import fast_tokenizer

base_dir = Path("./docs")
paths = list(base_dir.rglob("*"))


def load_docs(paths):
    all_docs = []
    for p in paths:
        if not p.is_file():
            continue

        ext = p.suffix.lower()
        try:
            if ext == ".html":
                # Try UnstructuredHTMLLoader first, fallback to BSHTMLLoader if it fails
                try:
                    loader = UnstructuredHTMLLoader(p)
                    docs = loader.load()    
                except (AttributeError, Exception) as e:
                    # Fallback to BSHTMLLoader for problematic HTML files
                    print(f"Warning: UnstructuredHTMLLoader failed for {p}, using BSHTMLLoader instead. Error: {type(e).__name__}")
                    loader = BSHTMLLoader(p)
                    docs = loader.load()
            elif ext == ".md":
                loader = UnstructuredMarkdownLoader(p)
                docs = loader.load()
            elif ext == ".txt":
                loader = TextLoader(p)
                docs = loader.load()
            else:
                print(f"Skipping {p} because it is not a supported file type")
                continue
            
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error loading {p}: {type(e).__name__}: {e}")
            continue
    
    return all_docs


# Temporary Python fallback for local Windows ingestion
def custom_token_length(text):
    # Ensure text is clean UTF-8
    clean_text = text.encode('utf-8', 'ignore').decode('utf-8')
    
    # A standard rule of thumb for English text is that 1 token is roughly 4 characters.
    # This avoids needing the C++ fast_tokenizer on Windows!
    return len(clean_text) // 4


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,       # Max 350 tokens per chunk
        chunk_overlap=50,     # Overlap of 50 tokens
        length_function=custom_token_length  # Tells LangChain to use your C++ tool
    )
    return splitter.split_documents(docs)


def store_docs(texts):
    print("Embedding documents and uploading to Pinecone... (This may take a minute)")
    
    model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = PineconeVectorStore.from_documents(
        documents=texts,
        embedding=model,
        index_name="rag-agent",
        pinecone_api_key=os.environ["PINECONE_API_KEY"],
    )
    return vectorstore


if __name__ == "__main__":
    docs = load_docs(paths)
    texts = split_docs(docs)
    vectorstore = store_docs(texts)

    print("="*50)
    print("✅ SUCCESS!")
    print(f"Documents Loaded: {len(docs)}")
    print(f"Total Chunks Uploaded to Pinecone: {len(texts)}")
    print("="*50)