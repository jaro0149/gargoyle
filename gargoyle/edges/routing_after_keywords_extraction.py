from langgraph.constants import END

from gargoyle.graph.mind_map_config import KeywordsMergingConfig
from gargoyle.graph.node_identifiers import ID_PREPARE_KEYWORDS_BEFORE_MERGING
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState


class RoutingAfterKeywordsExtraction:

    def __init__(self, config: KeywordsMergingConfig) -> None:
        self.config = config

    def __call__(self, state: AggregatedKeywordsState) -> str:
        if self.config.enabled:
            return ID_PREPARE_KEYWORDS_BEFORE_MERGING
        return END
