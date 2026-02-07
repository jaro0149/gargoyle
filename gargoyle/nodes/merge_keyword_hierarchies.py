from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from gargoyle.nodes.enforcing_utils import enforce_max_depth
from gargoyle.nodes.promt_templates import COMBINE_HIERARCHIES_PROMPT
from gargoyle.settings import KeywordsHierarchySettings, KeywordsMergingSetting
from gargoyle.state.aggregated_keywords_state import MergedKeywordsHierarchies
from gargoyle.state.keywords_state import RootKeywords, KeywordsHierarchy


class MergeKeywordHierarchies:

    def __init__(
            self,
            model: BaseChatModel,
            hierarchy_settings: KeywordsHierarchySettings,
            merge_settings: KeywordsMergingSetting
    ) -> None:
        self.hierarchy_settings = hierarchy_settings
        self.merge_settings = merge_settings
        self.struct_model = model.with_structured_output(schema=MergedKeywordsHierarchies)
        self.prompt = self._build_prompt(hierarchy_settings, merge_settings)

    @staticmethod
    def _build_prompt(hierarchy_settings: KeywordsHierarchySettings, merge_settings: KeywordsMergingSetting) -> str:
        return COMBINE_HIERARCHIES_PROMPT.format(
            max_depth=hierarchy_settings.max_depth,
            max_root_keywords=merge_settings.max_root_keywords
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

        trimmed_keywords = root_keywords.merged_keywords_hierarchies[:self.merge_settings.max_root_keywords]
        polished_hierarchies: list[KeywordsHierarchy] = []
        for hierarchy in trimmed_keywords:
            polished_hierarchy = enforce_max_depth(settings=self.hierarchy_settings, hierarchy=hierarchy)
            polished_hierarchies.append(polished_hierarchy)
        return MergedKeywordsHierarchies(merged_keywords_hierarchies=polished_hierarchies)
