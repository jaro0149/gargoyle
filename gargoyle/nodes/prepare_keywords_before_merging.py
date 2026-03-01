import random

from langgraph.runtime import Runtime
from langgraph.types import Overwrite

from gargoyle.graph.mind_map_context import MindMapContext
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState
from gargoyle.state.keywords_state import KeywordsHierarchy


def prepare_keywords_before_merging(
        state: AggregatedKeywordsState,
        runtime: Runtime[MindMapContext],
) -> AggregatedKeywordsState:
    """
    Prepare keywords by potentially randomizing and resetting their hierarchical structures before the merging process.

    :param state: Represents the current state of aggregated keywords including hierarchies
        and merged keyword information.
    :param runtime: Provides runtime context and configuration for the keyword merging process.
    :return: A new AggregatedKeywordsState object with the updated last keyword hierarchies
        based on the randomization setting and freshly reset merged keywords.
    """
    app_config = runtime.context.config.merge_keywords

    if state.merged_keywords_hierarchies:
        runtime.stream_writer("Using previously merged keyword hierarchies for preparation.")
        last_keyword_hierarchies = state.merged_keywords_hierarchies
    else:
        runtime.stream_writer("Using initial keyword hierarchies for preparation.")
        last_keyword_hierarchies = state.keyword_hierarchies

    if app_config.randomize_keywords:
        runtime.stream_writer("Randomizing keyword hierarchies order.")
        last_keyword_hierarchies = _shuffled_copy(last_keyword_hierarchies)

    runtime.stream_writer("Keywords prepared for merging.")
    return AggregatedKeywordsState(
        merged_keywords_hierarchies=Overwrite([]),
        last_keywords_hierarchies=last_keyword_hierarchies,
    )


def _shuffled_copy(items: list[KeywordsHierarchy]) -> list[KeywordsHierarchy]:
    return random.sample(list(items), k=len(items))
