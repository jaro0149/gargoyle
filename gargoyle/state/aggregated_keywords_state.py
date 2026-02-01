import operator
from typing import Annotated

from pydantic import BaseModel, ConfigDict

from gargoyle.state.keywords_state import KeywordsHierarchy


class MergedKeywordsHierarchies(BaseModel):
    """
    Merged keywords from multiple hierarchies.
    """

    merged_keywords_hierarchies: list[KeywordsHierarchy]
    """
    Merged root keyword hierarchies keywords from multiple hierarchies.
    """

    model_config = ConfigDict(extra="forbid")


class AggregatedKeywordsState(BaseModel):
    """
    Parent state containing a list of input texts and aggregated results.
    """

    input_texts: list[str] | None = None
    """
    A list of texts to process.
    """
    keyword_hierarchies: Annotated[list[KeywordsHierarchy], operator.add] = []
    """
    List of keyword hierarchies from all input texts.
    """
    merged_keywords_hierarchies: Annotated[list[KeywordsHierarchy], operator.add] = []
    """
    Merged keyword hierarchies from all input texts.
    """
