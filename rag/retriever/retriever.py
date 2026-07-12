import faiss, json, logging
from rag.embedding_model.embedding_model import load_model, encode

from rag.config import MODEL_NAME, INDEX_PATH, METADATA_PATH

from rag.exceptions import IndexNotFoundError, MetadataNotFoundError, EmptyRetrieverError

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, index_path, metadata_path, model_name):
        self.index = self._load_index(index_path)
        self.metadata = self._load_metadata(metadata_path)
        self.model = load_model(model_name)
    
    def _load_index(self, index_path):
        try:
            return faiss.read_index(index_path)
        except FileNotFoundError:
            logger.error('Path to index is not found')
            raise IndexNotFoundError(
                'Путь к index не найден: запустите build_index.'
            ) from None
    
    def _load_metadata(self, metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f) 
        except FileNotFoundError:
            logger.error('Path to metadata as not found')
            raise MetadataNotFoundError(
                'Путь к metadata не найдeн: запустите build_index'
            ) from None

    def retriever(self, user_query, top_k):
        logger.info(f'Search started | Query length: {len(user_query)} | Top_K: {top_k}')

        query_embedding = encode([user_query], self.model)
        faiss.normalize_L2(query_embedding)

        scores, ids = self.index.search(query_embedding, top_k)
    
        documents = [self.metadata[idx] for idx in ids[0]]

        if documents:
            top_score = float(scores[0][0])
            logger.info(f'Found {len(documents)} documents | Top score: {top_score:.4f}')

            return documents
        
        else:
            logger.warning('No documents found for the query')
            raise EmptyRetrieverError('Релевантные документы не были найдены')
        

    
