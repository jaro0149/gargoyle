
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from gargoyle.settings import settings


class LLMFactory:

    """Factory class for creating and managing LLM model instances."""

    def __init__(self) -> None:
        """Initialize the LLMFactory instance by building the LLM map."""
        self._llm_map: dict[str, BaseChatModel] = self._build_llm_map()

    @staticmethod
    def _build_llm_map() -> dict[str, BaseChatModel]:
        """Build a mapping of model IDs to ChatOpenAI instances based on settings."""
        llm_map: dict[str, BaseChatModel] = {}
        for model_id, model_settings in settings.llm_models.items():
            llm = ChatOpenAI(
                model=model_settings.model,
                api_key=model_settings.api_key.get_secret_value(),
                base_url=model_settings.base_url,
                streaming=True,
            )
            llm_map[model_id] = llm
        return llm_map

    def get_llm(self, model_id: str) -> BaseChatModel:
        """
        Retrieve a language model instance by its ID.

        :param model_id: The ID of the model to retrieve.
        :return: The requested model instance.
        :raises ValueError: If the requested model ID is not found in the configuration.
        """
        if model_id not in self._llm_map:
            msg = f"LLM model with ID '{model_id}' not found in configuration."
            raise ValueError(msg)
        return self._llm_map[model_id]
