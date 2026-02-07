from langgraph.constants import END
from langgraph.types import Send

from gargoyle.graph.mind_map_config import KeywordsMergingConfig
from gargoyle.graph.node_identifiers import ID_MERGE_HIERARCHIES
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState
from gargoyle.state.keywords_state import RootKeywords


class FanOutKeywordsMerging:

    def __init__(self, config: KeywordsMergingConfig) -> None:
        self.config = config

    def __call__(self, state: AggregatedKeywordsState) -> list[Send] | str:
        if not state.last_keywords_hierarchies:
            return END
        if len(state.last_keywords_hierarchies) <= self.config.max_root_keywords:
            return END

        buckets: list[Send] = []
        for hierarchy_idx in range(0, len(state.last_keywords_hierarchies), self.config.squash_root_keywords):
            chunk = state.keyword_hierarchies[hierarchy_idx:hierarchy_idx + self.config.squash_root_keywords]
            buckets.append(
                Send(
                    ID_MERGE_HIERARCHIES,
                    RootKeywords(keyword_hierarchies=chunk)
                )
            )
        return buckets
