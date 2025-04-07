import logging

from injector import singleton
from llama_index.core.llms import LLM, MockLLM

from readai.core.config import settings
from readai.core.schemas import LLMmode

logger = logging.getLogger(__name__)


@singleton
class LLMComponent:
    llm: LLM

    def __init__(self, llm_mode: LLMmode) -> None:
        match llm_mode:
            case LLMmode.DEEPSEEK:
                from llama_index.llms.deepseek import DeepSeek  # type: ignore

                self.llm = DeepSeek(
                    model=settings.DEEPSEEK_MODEL_NAME,
                    api_key=settings.DEEPSEEK_API_KEY,
                )

            case LLMmode.OPENAI:
                from llama_index.llms.openai import OpenAI  # type: ignore

                # openai_settings = settings.openai
                self.llm = OpenAI(
                    api_base="https://api.openai.com/v1",
                    api_key=settings.OPENAI_API_KEY,
                    model=settings.OPENAI_MODEL_NAME,
                )
            case LLMmode.OPENAI_LIKE:
                try:
                    from llama_index.llms.openai_like import OpenAILike  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "OpenAILike dependencies not found, install with `poetry install --extras llms-openai-like`"
                    ) from e

                # openai_settings = settings.openai
                self.llm = OpenAILike(
                    api_base="https://api.openai.com/v1",
                    api_key=settings.OPENAI_API_KEY,
                    model=settings.OPENAI_MODEL_NAME,
                )

            case LLMmode.MOCK_LLM:
                self.llm = MockLLM()
