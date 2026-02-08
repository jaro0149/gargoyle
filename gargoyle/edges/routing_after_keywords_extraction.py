from langgraph.runtime import Runtime

from gargoyle.graph.mind_map_context import MindMapContext
from gargoyle.graph.node_identifiers import ID_PREPARE_KEYWORDS_BEFORE_MERGING, ID_BUILD_MIND_MAP
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState


def route_after_keywords_extraction(_: AggregatedKeywordsState, runtime: Runtime[MindMapContext]) -> str:
    if runtime.context.config.merge_keywords.enabled:
        return ID_PREPARE_KEYWORDS_BEFORE_MERGING
    return ID_BUILD_MIND_MAP
