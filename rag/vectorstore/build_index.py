import json, faiss
from rag.prepare_data.prepare_data import prepare_dataset
from rag.embedding_model.embedding_model import load_model, encode
from rag.vectorstore.faiss_store import FaissStore

from rag.config import MODEL_NAME, DATASET_NAME, INDEX_PATH, METADATA_PATH

def build_index(dataset_name, model_name, index_path, metadata_path):
    dataset = prepare_dataset(dataset_name)
    model = load_model(model_name)

    dataset = dataset.map(
        lambda raw: {'document' : f"{raw['problem']}\n\n{raw['solution']}"}
    )

    embeddings = encode(dataset['document'], model)
    d = embeddings.shape[1]

    index = FaissStore(d)

    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    metadata = list(dataset['document'])

    index.save(index_path)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

build_index(DATASET_NAME, MODEL_NAME, INDEX_PATH, METADATA_PATH)


