from langgraph.constants import END
from langgraph.types import Send

from gargoyle.settings import KeywordsMergingSetting
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState
from gargoyle.state.keywords_state import RootKeywords


class FanOutKeywordsMerging:

    def __init__(self, settings: KeywordsMergingSetting) -> None:
        self.settings = settings

    def __call__(self, state: AggregatedKeywordsState) -> list[Send] | str:
        if not state.keyword_hierarchies:
            return END
        if len(state.keyword_hierarchies) <= self.settings.max_root_keywords:
            return END

        buckets: list[Send] = []
        for hierarchy_idx in range(0, len(state.keyword_hierarchies), self.settings.squash_root_keywords):
            chunk = state.keyword_hierarchies[hierarchy_idx:hierarchy_idx + self.settings.squash_root_keywords]
            buckets.append(
                Send(
                    "merge_hierarchies",
                    RootKeywords(keyword_hierarchies=chunk)
                )
            )
        return buckets
