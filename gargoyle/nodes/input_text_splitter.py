from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.runtime import Runtime

from gargoyle.graph.mind_map_context import MindMapContext
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState


def split_text(state: AggregatedKeywordsState, runtime: Runtime[MindMapContext]) -> AggregatedKeywordsState:
    if not state.text:
        return AggregatedKeywordsState(text_chunks=[])

    splitter_config = runtime.context.config.text_splitter
    if not splitter_config.enabled:
        return AggregatedKeywordsState(text_chunks=[state.text])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=splitter_config.chunk_size,
        chunk_overlap=splitter_config.chunk_overlap,
    )

    chunks = splitter.split_text(state.text)
    return AggregatedKeywordsState(
        text_chunks=chunks
    )
