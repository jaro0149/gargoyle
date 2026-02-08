from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime

from gargoyle.graph.mind_map_config import KeywordsExtractorConfig
from gargoyle.graph.mind_map_context import MindMapContext
from gargoyle.nodes.enforcing_utils import trim_keywords
from gargoyle.nodes.promt_templates import KEYWORDS_EXTRACTION_PROMPT
from gargoyle.state.keywords_state import Keywords, KeywordsState


class KeywordsExtractor:
    """
    Facilitates keyword extraction using a structured-output model.

    This class is designed to process input text and extract relevant keywords based on a
    structured-output model adhering to a predefined schema. It integrates with a base chat
    model and operates within the context of an application's state and runtime environment.
    """

    def __init__(self, model: BaseChatModel) -> None:
        """
        Initializes the instance with a structured-output model based on the provided base chat model.

        :param model: An instance of BaseChatModel that will be set up to output
            structured data adhering to the schema defined by the Keywords class.
        """
        self.struct_model = model.with_structured_output(schema=Keywords)

    def __call__(self, state: KeywordsState, runtime: Runtime[MindMapContext]) -> Keywords:
        """
        Processes input text within a given state and runtime to extract and refine keywords.

        The method uses a structured model to generate keywords based on a prompt and
        the input text. If no input text is provided, an empty list of keywords is returned.
        It also trims the derived keywords based on the application's configuration.

        :param state: The state object containing the input text and relevant context
            for keyword extraction.
        :param runtime: The runtime environment containing the context as well as the
            application's configuration.
        :return: A Keywords object containing the refined list of extracted keywords.
        """
        if not state.input_text:
            return Keywords(keywords=[])

        app_config = runtime.context.config.keywords_extractor
        prompt = self._build_prompt(app_config)
        llm_response = self.struct_model.invoke(
            input=[
                SystemMessage(content=prompt),
                HumanMessage(content=state.input_text)
            ]
        )
        keywords = cast(Keywords, llm_response)
        return trim_keywords(config=app_config, derived_keywords=keywords)

    @staticmethod
    def _build_prompt(config: KeywordsExtractorConfig) -> str:
        return KEYWORDS_EXTRACTION_PROMPT.format(
            max_keywords=config.max_keywords,
            max_words_in_keyword=config.max_words_in_keyword
        )
