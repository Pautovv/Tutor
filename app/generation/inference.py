import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase, PreTrainedModel

from huggingface_hub.errors import RepositoryNotFoundError
from requests.exceptions import ConnectionError
from app.exceptions import ModelLoadingError, EmptyGeneratorError

logger = logging.getLogger(__name__)

class Generator:
    def __init__(self, model_name: str) -> None:
        self.tokenizer = self._load_tokenizer(model_name)
        self.model = self._load_model(model_name)
    
    def _load_tokenizer(self, model_name: str) -> PreTrainedTokenizerBase:
        try:
            return AutoTokenizer.from_pretrained(model_name)
        except RepositoryNotFoundError as e:
            logger.error(f'Model {model_name} not found on HuggingFace Hub')
            raise ModelLoadingError(
                f'Токенизатор модели {model_name} не был найден на HF Hub: проверьте название.'
            ) from e
        except ConnectionError as e:
            logger.error(f'Network error while downloading {model_name}')
            raise ModelLoadingError(
                f'Не удалось скачать токенизатор: проблемы с сетью.'
            ) from e
    
    def _load_model(self, model_name: str) -> PreTrainedModel:
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype='auto'
            )
        except RepositoryNotFoundError as e:
            logger.error(f'Model {model_name} not found on HuggingFace Hub')
            raise ModelLoadingError(
                f'Модель {model_name} не была найдена на HF Hub: проверьте название.'
            ) from e
        except ConnectionError as e:
            logger.error(f'Network error while downloading {model_name}')
            raise ModelLoadingError(
                f'Не удалось скачать модель: проблемы с сетью.'
            ) from e


    def generate(self, messages: list[dict[str, str]], max_new_tokens: int, temperature: float) -> str:
        logger.info('Generating response...')
        inputs = self.tokenizer.apply_chat_template(
	        messages,
	        add_generation_prompt=True,
	        tokenize=True,
	        return_dict=True,
	        return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        
        if response:
            logger.info('Response generated')
            return response
        else:
            logger.warning('Response is empty')
            raise EmptyGeneratorError('Сгенерированный ответ модели пустой.')