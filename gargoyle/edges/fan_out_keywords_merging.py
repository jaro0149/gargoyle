from langgraph.constants import END
from langgraph.runtime import Runtime
from langgraph.types import Send

from gargoyle.graph.mind_map_context import MindMapContext
from gargoyle.graph.node_identifiers import ID_BUILD_MIND_MAP, ID_MERGE_HIERARCHIES
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState
from gargoyle.state.keywords_state import RootKeywords


def fan_out_merging_of_keywords(state: AggregatedKeywordsState, runtime: Runtime[MindMapContext]) -> list[Send] | str:
    """
    Perform a fan-out merging operation on keywords hierarchies by splitting them into manageable chunks.

    If the number of hierarchies is below a certain threshold, it triggers a specific operation instead of chunking.

    :param state: Represents the current state of aggregated keywords. Contains all the necessary hierarchy and states
        required for processing.
    :param runtime: Contains runtime execution context, including configurations for merging keywords and associated
        context information.
    :return: A list of `Send` objects representing chunks of keyword hierarchies ready for further processing, or a
        string ID indicating the appropriate action to perform based on the application configuration.
    """
    app_config = runtime.context.config.merge_keywords

    if not state.last_keywords_hierarchies:
        runtime.stream_writer("No keyword hierarchies found for merging.")
        return END
    if len(state.last_keywords_hierarchies) <= app_config.max_root_keywords:
        runtime.stream_writer(f"Number of hierarchies ({len(state.last_keywords_hierarchies)}) is less than"
                              f" or equal to merging threshold ({app_config.max_root_keywords})."
                              f" Proceeding to mind map building.")
        return ID_BUILD_MIND_MAP

    runtime.stream_writer(f"Fanning out keyword merging for {len(state.last_keywords_hierarchies)} hierarchies "
                          f"in chunks of {app_config.squash_root_keywords}.")
    buckets: list[Send] = []
    for hierarchy_idx in range(0, len(state.last_keywords_hierarchies), app_config.squash_root_keywords):
        chunk = state.keyword_hierarchies[hierarchy_idx:hierarchy_idx + app_config.squash_root_keywords]
        buckets.append(
            Send(
                ID_MERGE_HIERARCHIES,
                RootKeywords(keyword_hierarchies=chunk),
            ),
        )
    return buckets
