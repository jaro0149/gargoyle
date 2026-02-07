from langgraph.constants import END

from gargoyle.graph.mind_map_config import KeywordsMergingConfig
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState


class RoutingAfterKeywordsExtraction:

    def __init__(self, config: KeywordsMergingConfig) -> None:
        self.config = config

    def __call__(self, state: AggregatedKeywordsState) -> str:
        if self.config.enabled:
            return "prepare_keywords_before_merging"
        return END
