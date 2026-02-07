from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime

from gargoyle.graph.mind_map_config import KeywordsHierarchyConfig
from gargoyle.graph.mind_map_context import MindMapContext
from gargoyle.nodes.enforcing_utils import enforce_max_depth
from gargoyle.nodes.promt_templates import KEYWORDS_HIERARCHY_CREATION_PROMPT
from gargoyle.state.keywords_state import KeywordsState, RootKeywords, KeywordsHierarchy


class KeywordsHierarchyBuilder:

    def __init__(self, model: BaseChatModel) -> None:
        self.struct_model = model.with_structured_output(schema=RootKeywords)

    def __call__(self, state: KeywordsState, runtime: Runtime[MindMapContext]) -> RootKeywords:
        if not state.keywords:
            return RootKeywords(keyword_hierarchies=[])

        app_config = runtime.context.config.keywords_hierarchy
        prompt = self._build_prompt(app_config)
        llm_response = self.struct_model.invoke(
            input=[
                SystemMessage(content=prompt),
                HumanMessage(content=str(state.keywords))
            ]
        )
        root_keywords = cast(RootKeywords, llm_response)
        return self._enforce_constraints(root_keywords, app_config)

    @staticmethod
    def _build_prompt(config: KeywordsHierarchyConfig) -> str:
        return KEYWORDS_HIERARCHY_CREATION_PROMPT.format(
            max_depth=config.max_depth
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
