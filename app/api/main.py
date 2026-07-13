import logging
from logger_config import setup_logging

from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager

from app.api.schemas import UserRequest, ModelResponse

from rag.retriever.retriever import Retriever
from app.generation.inference import Generator
from app.pipeline import generate_answer

from rag.config import INDEX_PATH, METADATA_PATH, MODEL_NAME as EMBEDDING_MODEL
from app.config import MODEL_NAME 

from rag.exceptions import EmptyRetrieverError
from app.exceptions import EmptyGeneratorError


logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info('Server started')
    app.state.retriever = Retriever(INDEX_PATH, METADATA_PATH, EMBEDDING_MODEL)
    app.state.generator = Generator(MODEL_NAME)
    yield
    logger.info('Server ended')


app = FastAPI(lifespan=lifespan)

@app.post('/predict')
def model_response(dto: UserRequest, req: Request) -> ModelResponse:
    try: 
        response = generate_answer(
            dto.user_request,
            req.app.state.retriever,
            req.app.state.generator,
        )
    except EmptyGeneratorError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except EmptyRetrieverError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return ModelResponse(model_answer=response)

