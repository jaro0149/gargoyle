import random

from langgraph.runtime import Runtime
from langgraph.types import Overwrite

from gargoyle.graph.mind_map_context import MindMapContext
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState
from gargoyle.state.keywords_state import KeywordsHierarchy


def prepare_keywords_before_merging(
        state: AggregatedKeywordsState,
        runtime: Runtime[MindMapContext]
) -> AggregatedKeywordsState:
    app_config = runtime.context.config.merge_keywords

    if state.merged_keywords_hierarchies:
        last_keyword_hierarchies = state.merged_keywords_hierarchies
    else:
        last_keyword_hierarchies = state.keyword_hierarchies

    if app_config.randomize_keywords:
        last_keyword_hierarchies = _shuffled_copy(last_keyword_hierarchies)

    return AggregatedKeywordsState(
        merged_keywords_hierarchies=Overwrite([]),
        last_keywords_hierarchies=last_keyword_hierarchies
    )


def _shuffled_copy(items: list[KeywordsHierarchy]) -> list[KeywordsHierarchy]:
    return random.sample(list(items), k=len(items))
