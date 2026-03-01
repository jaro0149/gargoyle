from langgraph.runtime import Runtime

from gargoyle.graph.mind_map_context import MindMapContext
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState
from gargoyle.state.keywords_state import KeywordsHierarchy


def build_mind_map(state: AggregatedKeywordsState, runtime: Runtime[MindMapContext]) -> AggregatedKeywordsState:
    """
    Generate a PlantUML mind map representation based on the provided keywords hierarchies from the given state.

    :param state: The current state containing the keywords hierarchies and other
        related data for building the mind map.
    :param runtime: The runtime environment containing the context and configurations.
    :return: A new AggregatedKeywordsState object with an updated `mind_map_puml` containing the generated PlantUML
        mind map or `None` if no hierarchies exist.
    """
    keywords_hierarchies = state.last_keywords_hierarchies or state.keyword_hierarchies
    if not keywords_hierarchies:
        runtime.stream_writer("No keywords hierarchies to build mind map.")
        return AggregatedKeywordsState(mind_map_puml=None)

    runtime.stream_writer(f"Building mind map from {len(keywords_hierarchies)} hierarchies.")
    puml_lines = ["@startmindmap"]
    for hierarchy in keywords_hierarchies:
        puml_lines.extend(_build_puml_lines(hierarchy, 1))
    puml_lines.append("@endmindmap")

    puml_result = "\n".join(puml_lines)
    runtime.stream_writer("PlantUML mind map generated.")
    return AggregatedKeywordsState(
        mind_map_puml=puml_result,
    )


def _build_puml_lines(hierarchy: KeywordsHierarchy, depth: int) -> list[str]:
    """Recursively build PlantUML mind map lines."""
    prefix = "*" * depth
    lines = [f"{prefix} {hierarchy.keyword}"]
    if hierarchy.sub_keywords:
        for sub in hierarchy.sub_keywords:
            lines.extend(_build_puml_lines(sub, depth + 1))
    return lines
