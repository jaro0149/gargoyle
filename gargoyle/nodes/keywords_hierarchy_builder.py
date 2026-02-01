from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from gargoyle.nodes.promt_templates import KEYWORDS_HIERARCHY_CREATION_PROMPT
from gargoyle.nodes.utils import enforce_max_depth
from gargoyle.settings import KeywordsHierarchySettings
from gargoyle.state.keywords_state import KeywordsState, RootKeywords, KeywordsHierarchy


class KeywordsHierarchyBuilder:

    def __init__(self, model: BaseChatModel, settings: KeywordsHierarchySettings) -> None:
        self.settings = settings
        self.struct_model = model.with_structured_output(schema=RootKeywords)
        self.prompt = self._build_prompt(settings)

    @staticmethod
    def _build_prompt(settings: KeywordsHierarchySettings) -> str:
        return KEYWORDS_HIERARCHY_CREATION_PROMPT.format(
            max_depth=settings.max_depth
        )

    def __call__(self, state: KeywordsState) -> RootKeywords:
        if not state.keywords:
            return RootKeywords(keyword_hierarchies=[])

        llm_response = self.struct_model.invoke(
            input=[
                SystemMessage(content=self.prompt),
                HumanMessage(content=str(state.keywords))
            ]
        )
        root_keywords = cast(RootKeywords, llm_response)
        return self._enforce_constraints(root_keywords)

    def _enforce_constraints(self, root_keywords: RootKeywords) -> RootKeywords:
        if not root_keywords.keyword_hierarchies:
            return root_keywords
        polished_hierarchies: list[KeywordsHierarchy] = []

        for root_keyword in root_keywords.keyword_hierarchies:
            polished_hierarchy = enforce_max_depth(settings=self.settings, hierarchy=root_keyword)
            polished_hierarchies.append(polished_hierarchy)
        return RootKeywords(keyword_hierarchies=polished_hierarchies)
