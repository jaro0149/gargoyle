from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from gargoyle.graph.mind_map_config import MindMapConfig
from gargoyle.graph.mind_map_context import MindMapContext
from gargoyle.nodes.enforcing_utils import enforce_max_depth
from gargoyle.nodes.promt_templates import KEYWORDS_SINGLE_STEP_PROMPT
from gargoyle.state.keywords_state import KeywordsHierarchy, KeywordsState, RootKeywords

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable


class KeywordsSingleStepBuilder:

    """Combine keyword extraction and hierarchy building into a single LLM call."""

    def __init__(self, model: BaseChatModel) -> None:
        """
        Initialize the instance with a structured-output model.

        :param model: An instance of BaseChatModel.
        """
        self.struct_model: Runnable[Any, Any] = model.with_structured_output(schema=RootKeywords)  # type: ignore[reportUnknownMemberType]

    def __call__(self, state: KeywordsState, runtime: Runtime[MindMapContext]) -> RootKeywords:
        """
        Process input text to extract and organize keywords into a hierarchy in one step.

        :param state: The current state containing the input text.
        :param runtime: Runtime containing relevant context and configurations.
        :return: A RootKeywords object with hierarchical keywords.
        """
        if not state.input_text:
            runtime.stream_writer("No input text for single-step keyword hierarchy builder.")
            return RootKeywords(keyword_hierarchies=[])

        runtime.stream_writer(f"Extracting and building hierarchy from text "
                              f"(length: {len(state.input_text)} characters).")
        config = runtime.context.config
        prompt = self._build_prompt(config)
        llm_response = self.struct_model.invoke(
            input=[
                SystemMessage(content=prompt),
                HumanMessage(content=state.input_text),
            ],
        )
        if not isinstance(llm_response, RootKeywords):
            msg = f"Expected RootKeywords, got {type(llm_response)}"
            raise TypeError(msg)

        result = self._enforce_constraints(llm_response, config)

        if result.keyword_hierarchies:
            tree_strings = [h.to_string() for h in result.keyword_hierarchies]
            runtime.stream_writer("Built keyword hierarchy (single-step):\n" + "\n".join(tree_strings))
        else:
            runtime.stream_writer("No hierarchy was built.")

        return result

    @staticmethod
    def _build_prompt(config: MindMapConfig) -> str:
        return KEYWORDS_SINGLE_STEP_PROMPT.format(
            max_keywords=config.keywords_extractor.max_keywords,
            max_words_in_keyword=config.keywords_extractor.max_words_in_keyword,
            max_depth=config.keywords_hierarchy.max_depth,
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
