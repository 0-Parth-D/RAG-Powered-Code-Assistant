from langchain_community.document_loaders import UnstructuredHTMLLoader, UnstructuredMarkdownLoader, TextLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
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

def custom_token_length(text):
    tokens = fast_tokenizer.tokenize(text)
    return len(tokens)

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,       # Max 350 tokens per chunk
        chunk_overlap=50,     # Overlap of 50 tokens
        length_function=custom_token_length  # Tells LangChain to use your C++ tool
    )
    return splitter.split_documents(docs)

def store_docs(texts):
    model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=model,
        persist_directory="chroma_db",
        collection_name="rag_code_assistant"
    )
    return vectorstore

docs = load_docs(paths)
texts = split_docs(docs)
vectorstore = store_docs(texts)

print("Documents Loaded: ", len(docs))