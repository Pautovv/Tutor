import faiss, json
from rag.embedding_model.embedding_model import load_model, encode

model = load_model('sentence-transformers/all-MiniLM-L6-v2')
index = faiss.read_index('rag/data/processed/index.faiss')
with open('rag/data/processed/metadata.json', 'r') as f:
    metadata = json.load(f) 

def retriever(user_query, top_k):
    query_embedding = encode([user_query], model)
    faiss.normalize_L2(query_embedding)

    scores, ids = index.search(query_embedding, top_k)

    documents = [metadata[idx] for idx in ids[0]]

    return documents

    
