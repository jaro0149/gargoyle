from langgraph.constants import END
from langgraph.types import Send

from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState
from gargoyle.state.keywords_state import KeywordsState


def fan_out_keywords_extraction(state: AggregatedKeywordsState) -> list[Send] | str:
    if not state.input_texts:
        return END

    return [Send("build_keywords_hierarchies", KeywordsState(input_text=text)) for text in state.input_texts]
