import logging

from injector import singleton
from llama_index.core.llms import LLM, MockLLM

from readai.core.schemas import LLMmode

logger = logging.getLogger(__name__)


@singleton
class LLMComponent:
    llm: LLM

    # TODO 通过依赖注入配置信息 Settings
    def __init__(self, llm_mode: LLMmode) -> None:
        match llm_mode:
            case LLMmode.DEEPSEEK:
                from llama_index.llms.deepseek import DeepSeek  # type: ignore

                self.llm = DeepSeek(
                    model="deepseek-chat",
                    api_key="sk-1234567890",
                )

            case LLMmode.OPENAI:
                from llama_index.llms.openai import OpenAI  # type: ignore

                # openai_settings = settings.openai
                self.llm = OpenAI(
                    api_base="https://api.openai.com/v1",
                    api_key="sk-1234567890",
                    model="gpt-3.5-turbo",
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
                    api_key="sk-1234567890",
                    model="gpt-3.5-turbo",
                )
            case LLMmode.OLLAMA:
                from llama_index.llms.ollama import Ollama  # type: ignore

                # calculate llm model. If not provided tag, it will be use latest
                # model_name = (
                #     ollama_settings.llm_model + ":latest"
                #     if ":" not in ollama_settings.llm_model
                #     else ollama_settings.llm_model
                # )

                llm = Ollama(
                    model="llama3.1",
                    base_url="http://localhost:11434",
                    temperature=0.5,
                    context_window=1024,
                    additional_kwargs={},
                    request_timeout=60,
                )

                # if ollama_settings.autopull_models:
                #     from private_gpt.utils.ollama import check_connection, pull_model

                #     if not check_connection(llm.client):
                #         raise ValueError(
                #             f"Failed to connect to Ollama, "
                #             f"check if Ollama server is running on {ollama_settings.api_base}"
                #         )
                #     pull_model(llm.client, model_name)
                self.llm = llm

            case LLMmode.MOCK_LLM:
                self.llm = MockLLM()
