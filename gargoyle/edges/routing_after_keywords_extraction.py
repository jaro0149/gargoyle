from langgraph.runtime import Runtime

from gargoyle.graph.mind_map_context import MindMapContext
from gargoyle.graph.node_identifiers import ID_PREPARE_KEYWORDS_BEFORE_MERGING, ID_BUILD_MIND_MAP
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState


def route_after_keywords_extraction(_: AggregatedKeywordsState, runtime: Runtime[MindMapContext]) -> str:
    """
    Determines the appropriate next route based on the current runtime configuration and aggregated keywords state.

    This function evaluates the runtime context configuration to decide whether to prepare keywords for merging
    or to proceed directly to building the mind map. It considers the `merge_keywords,enabled` flag in the context
    configuration to make the decision.

    :param runtime: The runtime instance containing the operational context and configuration settings
        for the mind map process.
    :return: The identifier corresponding to the next stage in the workflow. Returns `ID_PREPARE_KEYWORDS_BEFORE_MERGING`
        if keyword merging is enabled in the configuration, otherwise returns `ID_BUILD_MIND_MAP`.
    """
    if runtime.context.config.merge_keywords.enabled:
        return ID_PREPARE_KEYWORDS_BEFORE_MERGING
    return ID_BUILD_MIND_MAP
