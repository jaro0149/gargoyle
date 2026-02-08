import operator
from typing import Annotated

from langgraph.types import Overwrite
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
    merged_keywords_hierarchies: Annotated[list[KeywordsHierarchy], operator.add] | Overwrite = []
    """
    Merged keyword hierarchies from all input texts.
    This list is automatically erased before the next merging iteration and built at the end of the actual iteration.
    """
    last_keywords_hierarchies: list[KeywordsHierarchy] = []
    """
    List of keyword hierarchies from the last iteration of the merging process.
    """
    mind_map_puml: str | None = None
    """
    Built PlantUML representation of the mind map.
    """
