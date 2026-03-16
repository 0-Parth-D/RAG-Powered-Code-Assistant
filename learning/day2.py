from sentence_transformers import SentenceTransformer
import numpy as np
import hashlib
import chromadb

# Initialize the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize the Chroma client
client = chromadb.PersistentClient(path="chroma_db")

# Initialize the collection
collection = client.get_or_create_collection("test_collection", metadata={"hnsw:space": "cosine"})

# Function to embed and store the sentences in the collection
def embed_and_store(sentences, collection):
    embeddings = model.encode(sentences)
    ids = [hashlib.sha256(sentence.encode()).hexdigest() for sentence in sentences]
    collection.upsert(
        documents=sentences,
        ids=ids,
        embeddings=embeddings,
    )
    return ids

# Function to perform semantic search
def semantic_search(query, model=model, collection=collection, n_results=3):
    query_embedding = model.encode(query)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )
    return results

sentences = [
    "How do I read a file in Python?",
    "Open a file using the open() function",
    "Python file I/O tutorial",
    "What is a for loop?",
    "Iterating over a list in Python",
    "How to use list comprehensions",
    "Docker container vs Docker image",
    "What is a Dockerfile?",
    "Sort a list in Python",
    "The sky is blue and the sun is yellow"
]

ids = embed_and_store(sentences, collection)

user_query = "reading and writing files in Python"
results = semantic_search(user_query, model=model, collection=collection, n_results=3)
print(results)