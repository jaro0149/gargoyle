from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime

from gargoyle.graph.mind_map_config import MindMapConfig
from gargoyle.graph.mind_map_context import MindMapContext
from gargoyle.nodes.enforcing_utils import enforce_max_depth
from gargoyle.nodes.promt_templates import KEYWORDS_SINGLE_STEP_PROMPT
from gargoyle.state.keywords_state import KeywordsState, RootKeywords, KeywordsHierarchy


class KeywordsSingleStepBuilder:
    """
    Combines keyword extraction and hierarchy building into a single LLM call.
    """

    def __init__(self, model: BaseChatModel) -> None:
        """
        Initializes the instance with a structured-output model.

        :param model: An instance of BaseChatModel.
        """
        self.struct_model = model.with_structured_output(schema=RootKeywords)

    def __call__(self, state: KeywordsState, runtime: Runtime[MindMapContext]) -> RootKeywords:
        """
        Processes input text to extract and organize keywords into a hierarchy in one step.

        :param state: The current state containing the input text.
        :param runtime: Runtime containing relevant context and configurations.
        :return: A RootKeywords object with hierarchical keywords.
        """
        if not state.input_text:
            return RootKeywords(keyword_hierarchies=[])

        config = runtime.context.config
        prompt = self._build_prompt(config)
        llm_response = self.struct_model.invoke(
            input=[
                SystemMessage(content=prompt),
                HumanMessage(content=state.input_text)
            ]
        )
        root_keywords = cast(RootKeywords, llm_response)
        return self._enforce_constraints(root_keywords, config)

    @staticmethod
    def _build_prompt(config: MindMapConfig) -> str:
        return KEYWORDS_SINGLE_STEP_PROMPT.format(
            max_keywords=config.keywords_extractor.max_keywords,
            max_words_in_keyword=config.keywords_extractor.max_words_in_keyword,
            max_depth=config.keywords_hierarchy.max_depth
        )

    @staticmethod
    def _enforce_constraints(root_keywords: RootKeywords, config: MindMapConfig) -> RootKeywords:
        if not root_keywords.keyword_hierarchies:
            return root_keywords
        polished_hierarchies: list[KeywordsHierarchy] = []

        for root_keyword in root_keywords.keyword_hierarchies:
            polished_hierarchy = enforce_max_depth(config=config.keywords_hierarchy, hierarchy=root_keyword)
            polished_hierarchies.append(polished_hierarchy)
        return RootKeywords(keyword_hierarchies=polished_hierarchies)
