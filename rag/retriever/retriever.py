import faiss, json
from rag.embedding_model.embedding_model import load_model, encode

from rag.config import MODEL_NAME, INDEX_PATH, METADATA_PATH

model = load_model(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f) 

def retriever(user_query, top_k):
    query_embedding = encode([user_query], model)
    faiss.normalize_L2(query_embedding)

    scores, ids = index.search(query_embedding, top_k)

    documents = [metadata[idx] for idx in ids[0]]

    return documents

    
