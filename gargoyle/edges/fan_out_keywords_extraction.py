from langgraph.constants import END
from langgraph.types import Send

from gargoyle.graph.node_identifiers import ID_BUILD_KEYWORDS_HIERARCHIES
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState
from gargoyle.state.keywords_state import KeywordsState


def fan_out_keywords_extraction(state: AggregatedKeywordsState) -> list[Send] | str:
    """
    Extracts keywords from each text chunk in the provided state.

    This function takes the given `state` containing multiple text chunks and creates
    commands to process each chunk individually. If no text chunks are present in the
    state, a termination signal is returned.

    :param state: The current aggregation state containing extracted text chunks.
    :return: A list of commands for processing individual text chunks into keyword
        hierarchies, or a termination signal if no text chunks are provided.
    """
    if not state.text_chunks:
        return END

    return [Send(ID_BUILD_KEYWORDS_HIERARCHIES, KeywordsState(input_text=text)) for text in state.text_chunks]
