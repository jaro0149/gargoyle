from langgraph.runtime import Runtime

from gargoyle.graph.mind_map_context import MindMapContext
from gargoyle.graph.node_identifiers import ID_EXTRACT_KEYWORDS, ID_EXTRACT_KEYWORDS_SINGLE_STEP
from gargoyle.state.keywords_state import KeywordsState


def route_keywords_extraction(_: KeywordsState, runtime: Runtime[MindMapContext]) -> str:
    """
    Extract the appropriate routing ID for keyword extraction based on the configuration.

    This function determines which keyword extraction route ID to use depending on the runtime
    context configuration. If the configuration specifies the use of a single-step keyword
    extraction process, a specific route ID is returned. Otherwise, the default route ID for
    keyword extraction is returned.

    :param _: Unused state of the graph.
    :param runtime: The runtime instance containing the execution context for managing
        the mind map extraction process.
    :return: The routing ID for the keyword extraction process. This could be either the
        single-step extraction ID or a default extraction ID based on the active configuration.
    """
    if runtime.context.config.keywords_hierarchy.use_single_step:
        runtime.stream_writer("Routing to single-step keyword hierarchy builder.")
        return ID_EXTRACT_KEYWORDS_SINGLE_STEP
    runtime.stream_writer("Routing to multi-step keyword extraction and hierarchy builder.")
    return ID_EXTRACT_KEYWORDS
