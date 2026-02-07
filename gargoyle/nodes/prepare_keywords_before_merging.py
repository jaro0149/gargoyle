import random

from langgraph.types import Overwrite

from gargoyle.settings import KeywordsMergingSetting
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState
from gargoyle.state.keywords_state import KeywordsHierarchy


class PrepareKeywordsBeforeMerging:

    def __init__(self, settings: KeywordsMergingSetting) -> None:
        self.settings = settings

    def __call__(self, state: AggregatedKeywordsState) -> AggregatedKeywordsState:
        if state.merged_keywords_hierarchies:
            last_keyword_hierarchies = state.merged_keywords_hierarchies
        else:
            last_keyword_hierarchies = state.keyword_hierarchies

        if self.settings.randomize_keywords:
            last_keyword_hierarchies = self._shuffled_copy(last_keyword_hierarchies)

        return AggregatedKeywordsState(
            merged_keywords_hierarchies=Overwrite([]),
            last_keywords_hierarchies=last_keyword_hierarchies
        )

    @staticmethod
    def _shuffled_copy(items: list[KeywordsHierarchy]) -> list[KeywordsHierarchy]:
        return random.sample(list(items), k=len(items))
