from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from gargoyle.graph.mind_map_config import KeywordsHierarchyConfig, KeywordsMergingConfig
from gargoyle.graph.mind_map_context import MindMapContext
from gargoyle.nodes.enforcing_utils import enforce_max_depth
from gargoyle.nodes.promt_templates import COMBINE_HIERARCHIES_PROMPT
from gargoyle.state.aggregated_keywords_state import MergedKeywordsHierarchies
from gargoyle.state.keywords_state import RootKeywords

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable


class MergeKeywordHierarchies:

    """
    Handle the merging of keyword hierarchies based on structured configurations.

    This class is designed to process and merge keyword hierarchies, using a
    structured output schema and configurable constraints. It integrates with a
    language model to generate and enforce the resulting hierarchies.
    """

    def __init__(self, model: BaseChatModel) -> None:
        """
        Initialize an instance with a structured model based on the provided chat model.

        :param model: The base chat model to be configured into a structured model.
        """
        self.struct_model: Runnable[Any, Any] = model.with_structured_output(schema=MergedKeywordsHierarchies)  # type: ignore[reportUnknownMemberType]

    def __call__(self, state: RootKeywords, runtime: Runtime[MindMapContext]) -> MergedKeywordsHierarchies:
        """
        Execute the callable object to process keyword hierarchies and perform merging of keywords.

        This method constructs prompt messages, invokes the structured model with
        specific constraints, and processes the response to generate the resulting
        merged keyword hierarchies.

        :param state: The root keywords state containing existing keyword hierarchies.
        :param runtime: The runtime object responsible for providing context and configurations for processing.
        :return: The result containing merged keyword hierarchies processed using the given configurations.
        """
        if not state.keyword_hierarchies:
            runtime.stream_writer("No keyword hierarchies to merge.")
            return MergedKeywordsHierarchies(merged_keywords_hierarchies=[])

        runtime.stream_writer(f"Merging {len(state.keyword_hierarchies)} keyword hierarchies.")
        app_config = runtime.context.config
        prompt = self._build_prompt(app_config.keywords_hierarchy, app_config.merge_keywords)
        input_message = self._create_input_message(state)
        llm_response = self.struct_model.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=input_message),
            ],
        )
        if not isinstance(llm_response, MergedKeywordsHierarchies):
            msg = f"Expected MergedKeywordsHierarchies, got {type(llm_response)}"
            raise TypeError(msg)

        result = self._enforce_constraints(
            root_keywords=llm_response,
            hierarchy_config=app_config.keywords_hierarchy,
            merge_config=app_config.merge_keywords,
        )

        if result.merged_keywords_hierarchies:
            tree_strings = [h.to_string() for h in result.merged_keywords_hierarchies]
            runtime.stream_writer("Merged keyword hierarchy:\n" + "\n".join(tree_strings))
        else:
            runtime.stream_writer("No merged hierarchy was produced.")

        return result

    @staticmethod
    def _build_prompt(hierarchy_config: KeywordsHierarchyConfig, merge_config: KeywordsMergingConfig) -> str:
        return COMBINE_HIERARCHIES_PROMPT.format(
            max_depth=hierarchy_config.max_depth,
            max_root_keywords=merge_config.max_root_keywords,
        )

    @staticmethod
    def _create_input_message(state: RootKeywords) -> str:
        if not state.keyword_hierarchies:
            return ""
        input_texts = [keyword_hierarchy.to_string() for keyword_hierarchy in state.keyword_hierarchies]
        return "\n\n".join(input_texts)

    @staticmethod
    def _enforce_constraints(
            root_keywords: MergedKeywordsHierarchies,
            hierarchy_config: KeywordsHierarchyConfig,
            merge_config: KeywordsMergingConfig,
    ) -> MergedKeywordsHierarchies:
        if not root_keywords.merged_keywords_hierarchies:
            return root_keywords

        trimmed_keywords = root_keywords.merged_keywords_hierarchies[:merge_config.max_root_keywords]
        polished_hierarchies = [
            enforce_max_depth(config=hierarchy_config, hierarchy=hierarchy)
            for hierarchy in trimmed_keywords
        ]
        return MergedKeywordsHierarchies(merged_keywords_hierarchies=polished_hierarchies)
