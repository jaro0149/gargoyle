from langgraph.constants import END
from langgraph.runtime import Runtime
from langgraph.types import Send

from gargoyle.graph.mind_map_context import MindMapContext
from gargoyle.graph.node_identifiers import ID_BUILD_KEYWORDS_HIERARCHIES
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState
from gargoyle.state.keywords_state import KeywordsState


def fan_out_keywords_extraction(state: AggregatedKeywordsState, runtime: Runtime[MindMapContext]) -> list[Send] | str:
    """
    Extract keywords from each text chunk in the provided state.

    This function takes the given `state` containing multiple text chunks and creates
    commands to process each chunk individually. If no text chunks are present in the
    state, a termination signal is returned.

    :param state: The current aggregation state containing extracted text chunks.
    :param runtime: Provides runtime context for the extraction process.
    :return: A list of commands for processing individual text chunks into keyword
        hierarchies, or a termination signal if no text chunks are provided.
    """
    if not state.text_chunks:
        runtime.stream_writer("No text chunks found for fan-out extraction.")
        return END

    runtime.stream_writer(f"Fanning out keywords extraction for {len(state.text_chunks)} chunks.")
    return [Send(ID_BUILD_KEYWORDS_HIERARCHIES, KeywordsState(input_text=text)) for text in state.text_chunks]
