from langgraph.runtime import Runtime

from gargoyle.graph.mind_map_context import MindMapContext
from gargoyle.graph.node_identifiers import ID_BUILD_MIND_MAP, ID_PREPARE_KEYWORDS_BEFORE_MERGING
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState


def route_after_keywords_extraction(_: AggregatedKeywordsState, runtime: Runtime[MindMapContext]) -> str:
    """
    Determine the appropriate next route based on the current runtime configuration and aggregated keywords state.

    This function evaluates the runtime context configuration to decide whether to prepare keywords for merging
    or to proceed directly to building the mind map. It considers the `merge_keywords,enabled` flag in the context
    configuration to make the decision.

    :param _: Unused state of the graph.
    :param runtime: The runtime instance containing the operational context and configuration settings
        for the mind map process.
    :return: The identifier corresponding to the next stage in the workflow.
        Returns `ID_PREPARE_KEYWORDS_BEFORE_MERGING` if keyword merging is enabled in the configuration,
        otherwise returns `ID_BUILD_MIND_MAP`.
    """
    if runtime.context.config.merge_keywords.enabled:
        runtime.stream_writer("Keywords merging is enabled. Routing to keyword preparation.")
        return ID_PREPARE_KEYWORDS_BEFORE_MERGING
    runtime.stream_writer("Keywords merging is disabled. Routing directly to mind map building.")
    return ID_BUILD_MIND_MAP
