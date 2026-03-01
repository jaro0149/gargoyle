from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from gargoyle.graph.mind_map_config import KeywordsHierarchyConfig
from gargoyle.graph.mind_map_context import MindMapContext
from gargoyle.nodes.enforcing_utils import enforce_max_depth
from gargoyle.nodes.promt_templates import KEYWORDS_HIERARCHY_CREATION_PROMPT
from gargoyle.state.keywords_state import KeywordsHierarchy, KeywordsState, RootKeywords


class KeywordsHierarchyBuilder:

    """
    Build a hierarchical structure of keywords based on input models and configurations.

    This class is designed to facilitate the creation of keyword hierarchies by interacting
    with a BaseChatModel. It leverages structured data outputs defined by the RootKeywords
    schema. The purpose of this class is to process keywords, generate hierarchical prompts,
    and enforce structural constraints, ensuring the output conforms to pre-defined rules.
    """

    def __init__(self, model: BaseChatModel) -> None:
        """
        Initialize an instance of the class.

        :param model: Instance of the BaseChatModel that is configured for structured output usage.
        """
        self.struct_model = model.with_structured_output(schema=RootKeywords)

    def __call__(self, state: KeywordsState, runtime: Runtime[MindMapContext]) -> RootKeywords:
        """
        Process the state and runtime inputs to generate and return structured RootKeywords.

        This method constructs a prompt based on the application's configuration, invokes the
        language model to generate a structured response, and applies constraints to ensure
        the output adheres to predefined rules.

        :param state: The current state containing keywords.
        :param runtime: Runtime containing relevant context and configurations for execution.
        :return: A RootKeywords object resulting from processing the input state and
            applying configuration constraints.
        """
        if not state.keywords:
            runtime.stream_writer("No keywords to build hierarchy.")
            return RootKeywords(keyword_hierarchies=[])

        runtime.stream_writer(f"Building hierarchy from keywords: {state.keywords}")
        app_config = runtime.context.config.keywords_hierarchy
        prompt = self._build_prompt(app_config)
        llm_response = self.struct_model.invoke(
            input=[
                SystemMessage(content=prompt),
                HumanMessage(content=str(state.keywords)),
            ],
        )
        root_keywords = cast("RootKeywords", llm_response)
        result = self._enforce_constraints(root_keywords, app_config)

        if result.keyword_hierarchies:
            tree_strings = [h.to_string() for h in result.keyword_hierarchies]
            runtime.stream_writer("Built keyword hierarchy:\n" + "\n".join(tree_strings))
        else:
            runtime.stream_writer("No hierarchy was built.")

        return result

    @staticmethod
    def _build_prompt(config: KeywordsHierarchyConfig) -> str:
        return KEYWORDS_HIERARCHY_CREATION_PROMPT.format(
            max_depth=config.max_depth,
        )

    @staticmethod
    def _enforce_constraints(root_keywords: RootKeywords, config: KeywordsHierarchyConfig) -> RootKeywords:
        if not root_keywords.keyword_hierarchies:
            return root_keywords
        polished_hierarchies: list[KeywordsHierarchy] = []

        for root_keyword in root_keywords.keyword_hierarchies:
            polished_hierarchy = enforce_max_depth(config=config, hierarchy=root_keyword)
            polished_hierarchies.append(polished_hierarchy)
        return RootKeywords(keyword_hierarchies=polished_hierarchies)
