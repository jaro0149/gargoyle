from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from gargoyle.nodes.enforcing_utils import trim_keywords
from gargoyle.nodes.promt_templates import KEYWORDS_EXTRACTION_PROMPT
from gargoyle.settings import KeywordExtractorSettings
from gargoyle.state.keywords_state import Keywords, KeywordsState


class KeywordsExtractor:

    def __init__(self, model: BaseChatModel, settings: KeywordExtractorSettings) -> None:
        self.struct_model = model.with_structured_output(schema=Keywords)
        self.prompt = self._build_prompt(settings)
        self.settings = settings

    @staticmethod
    def _build_prompt(settings: KeywordExtractorSettings) -> str:
        return KEYWORDS_EXTRACTION_PROMPT.format(
            max_keywords=settings.max_keywords,
            max_words_in_keyword=settings.max_words_in_keyword
        )

    def __call__(self, state: KeywordsState) -> Keywords:
        if not state.input_text:
            return Keywords(keywords=[])

        llm_response = self.struct_model.invoke(
            input=[
                SystemMessage(content=self.prompt),
                HumanMessage(content=state.input_text)
            ]
        )
        keywords = cast(Keywords, llm_response)
        return trim_keywords(settings=self.settings, derived_keywords=keywords)
