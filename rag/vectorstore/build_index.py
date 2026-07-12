import json, faiss, logging
from rag.prepare_data.prepare_data import prepare_dataset
from rag.embedding_model.embedding_model import load_model, encode
from rag.vectorstore.faiss_store import FaissStore

from rag.config import MODEL_NAME, DATASET_NAME, INDEX_PATH, METADATA_PATH
from logger_config import setup_logging

from rag.exceptions import DatasetParsingError

logger = logging.getLogger(__name__)

def build_index(dataset_name: str, model_name: str, index_path: str, metadata_path: str) -> None:
    logger.info('Starting index building')
    dataset = prepare_dataset(dataset_name)

    logger.info(f'Loading model: {model_name}...')
    model = load_model(model_name)
    logger.info(f'Model loaded successfully')

    logger.info('Creating combined documents...')
    dataset = dataset.map(
        lambda raw: {'document' : f"{raw['problem']}\n\n{raw['solution']}"}
    )
    if len(dataset)==0:
        logger.error('Dataset became empty after combined documents')
        raise DatasetParsingError(
            'Датасет пустой: проверить функцию создания документов.'
        )
    else:
        logger.info(f'Created {len(dataset)} documents')


    logger.info('Computing embeddings...')
    embeddings = encode(dataset['document'], model)
    logger.info(f'Embeddings computed. Shape: {embeddings.shape}')
    d = embeddings.shape[1]

    logger.info(f'Building FAISS-index with dimension {d}...')
    index = FaissStore(d)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    logger.info(f'Index builded successfully')

    metadata = list(dataset['document'])

    logger.info('Saving index and metadata...')
    index.save(index_path)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f)
    logger.info(f'Index and metadata saved\nIndex -> {index_path}\nMetadata -> {metadata_path}')
    
    logger.info('Index building completed successfully')

if __name__ == '__main__':
    setup_logging()
    build_index(DATASET_NAME, MODEL_NAME, INDEX_PATH, METADATA_PATH)


