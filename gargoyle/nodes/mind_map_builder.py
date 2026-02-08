from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState
from gargoyle.state.keywords_state import KeywordsHierarchy


def build_mind_map(state: AggregatedKeywordsState) -> AggregatedKeywordsState:
    """
    Generates a PlantUML mind map representation based on the provided keywords hierarchies from the given state.

    :param state: The current state containing the keywords hierarchies and other
        related data for building the mind map.
    :return: A new AggregatedKeywordsState object with an updated `mind_map_puml` containing the generated PlantUML
        mind map or `None` if no hierarchies exist.
    """
    keywords_hierarchies = state.last_keywords_hierarchies or state.keyword_hierarchies
    if not keywords_hierarchies:
        return AggregatedKeywordsState(mind_map_puml=None)

    puml_lines = ["@startmindmap"]
    for hierarchy in keywords_hierarchies:
        puml_lines.extend(_build_puml_lines(hierarchy, 1))
    puml_lines.append("@endmindmap")

    return AggregatedKeywordsState(
        mind_map_puml="\n".join(puml_lines)
    )


def _build_puml_lines(hierarchy: KeywordsHierarchy, depth: int) -> list[str]:
    """
    Recursively build PlantUML mind map lines.
    """
    prefix = "*" * depth
    lines = [f"{prefix} {hierarchy.keyword}"]
    if hierarchy.sub_keywords:
        for sub in hierarchy.sub_keywords:
            lines.extend(_build_puml_lines(sub, depth + 1))
    return lines
