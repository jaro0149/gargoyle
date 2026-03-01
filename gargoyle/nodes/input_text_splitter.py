from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.runtime import Runtime

from gargoyle.graph.mind_map_context import MindMapContext
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState


def split_text(state: AggregatedKeywordsState, runtime: Runtime[MindMapContext]) -> AggregatedKeywordsState:
    """
    Split the given text into smaller chunks based on the configuration provided.

    This function takes the current state containing the input text and splits it into
    smaller chunks according to the text splitter's configuration present in the runtime
    context. If the text splitter is disabled in the configuration, the function will return
    the original text as a single chunk. Otherwise, it uses a recursive character splitter
    to generate the chunks.

    :param state: The current aggregated state, which includes the input text to be split.
    :param runtime: The runtime environment containing the configuration required for
        splitting text. The context of the runtime provides access to the configuration.
    :return: A new state containing the list of text chunks created from the input text.
    """
    if not state.text:
        runtime.stream_writer("No input text to split.")
        return AggregatedKeywordsState(text_chunks=[])

    splitter_config = runtime.context.config.text_splitter
    if not splitter_config.enabled:
        runtime.stream_writer("Text splitter is disabled. Using the entire text as a single chunk.")
        return AggregatedKeywordsState(text_chunks=[state.text])

    runtime.stream_writer(f"Splitting text into chunks with size {splitter_config.chunk_size} "
                          f"and overlap {splitter_config.chunk_overlap} characters.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=splitter_config.chunk_size,
        chunk_overlap=splitter_config.chunk_overlap,
    )

    chunks = splitter.split_text(state.text)
    runtime.stream_writer(f"Text split into {len(chunks)} chunks.")
    return AggregatedKeywordsState(
        text_chunks=chunks,
    )
