from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from gargoyle.nodes.enforcing_utils import enforce_max_depth
from gargoyle.nodes.promt_templates import COMBINE_HIERARCHIES_PROMPT
from gargoyle.graph.mind_map_config import KeywordsHierarchyConfig, KeywordsMergingConfig
from gargoyle.state.aggregated_keywords_state import MergedKeywordsHierarchies
from gargoyle.state.keywords_state import RootKeywords, KeywordsHierarchy


class MergeKeywordHierarchies:

    def __init__(
            self,
            model: BaseChatModel,
            hierarchy_config: KeywordsHierarchyConfig,
            merge_config: KeywordsMergingConfig
    ) -> None:
        self.hierarchy_config = hierarchy_config
        self.merge_config = merge_config
        self.struct_model = model.with_structured_output(schema=MergedKeywordsHierarchies)
        self.prompt = self._build_prompt(hierarchy_config, merge_config)

    @staticmethod
    def _build_prompt(hierarchy_config: KeywordsHierarchyConfig, merge_config: KeywordsMergingConfig) -> str:
        return COMBINE_HIERARCHIES_PROMPT.format(
            max_depth=hierarchy_config.max_depth,
            max_root_keywords=merge_config.max_root_keywords
        )

    def __call__(self, state: RootKeywords) -> MergedKeywordsHierarchies:
        if not state.keyword_hierarchies:
            return MergedKeywordsHierarchies(merged_keywords_hierarchies=[])

        input_message = self._create_input_message(state)
        llm_response = self.struct_model.invoke(
            [
                SystemMessage(content=self.prompt),
                HumanMessage(content=input_message)
            ]
        )
        root_keywords = cast(MergedKeywordsHierarchies, llm_response)
        return self._enforce_constraints(root_keywords=root_keywords)

    @staticmethod
    def _create_input_message(state: RootKeywords) -> str:
        input_texts = []
        for keyword_hierarchy in state.keyword_hierarchies:
            input_texts.append(keyword_hierarchy.to_string())
        return "\n\n".join(input_texts)

    def _enforce_constraints(self, root_keywords: MergedKeywordsHierarchies) -> MergedKeywordsHierarchies:
        if not root_keywords.merged_keywords_hierarchies:
            return root_keywords

        trimmed_keywords = root_keywords.merged_keywords_hierarchies[:self.merge_config.max_root_keywords]
        polished_hierarchies: list[KeywordsHierarchy] = []
        for hierarchy in trimmed_keywords:
            polished_hierarchy = enforce_max_depth(config=self.hierarchy_config, hierarchy=hierarchy)
            polished_hierarchies.append(polished_hierarchy)
        return MergedKeywordsHierarchies(merged_keywords_hierarchies=polished_hierarchies)
