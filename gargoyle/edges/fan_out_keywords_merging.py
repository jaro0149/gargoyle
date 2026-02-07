from langgraph.constants import END
from langgraph.runtime import Runtime
from langgraph.types import Send

from gargoyle.graph.mind_map_context import MindMapContext
from gargoyle.graph.node_identifiers import ID_MERGE_HIERARCHIES
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState
from gargoyle.state.keywords_state import RootKeywords


def fan_out_merging_of_keywords(state: AggregatedKeywordsState, runtime: Runtime[MindMapContext]) -> list[Send] | str:
    app_config = runtime.context.config.merge_keywords

    if not state.last_keywords_hierarchies:
        return END
    if len(state.last_keywords_hierarchies) <= app_config.max_root_keywords:
        return END

    buckets: list[Send] = []
    for hierarchy_idx in range(0, len(state.last_keywords_hierarchies), app_config.squash_root_keywords):
        chunk = state.keyword_hierarchies[hierarchy_idx:hierarchy_idx + app_config.squash_root_keywords]
        buckets.append(
            Send(
                ID_MERGE_HIERARCHIES,
                RootKeywords(keyword_hierarchies=chunk)
            )
        )
    return buckets
