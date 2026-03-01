from pydantic import BaseModel, ConfigDict


class Keywords(BaseModel):

    """Schema for extracting keywords from text."""

    keywords: list[str] | None = None
    """
    A list of keywords extracted from the text.
    """

    model_config = ConfigDict(extra="forbid")


class KeywordsHierarchy(BaseModel):

    """Schema for representing a keyword and its sub-keywords."""

    keyword: str
    """
    A keyword extracted from the text.
    """
    sub_keywords: list["KeywordsHierarchy"] | None
    """
    A list of sub-keywords related to the main keyword.
    """

    model_config = ConfigDict(extra="forbid")

    def to_string(self) -> str:
        """
        Create a pretty-printed string representation of the tree structure.

        :return: A string representation of the tree structure.
        """
        return self._build_tree_string([self], 0)

    def _build_tree_string(self, nodes: list["KeywordsHierarchy"] | None, depth: int) -> str:
        if not nodes:
            return ""
        tree_str = ""
        for node in nodes:
            tree_str += "  " * depth + f"- {node.keyword}\n"
            tree_str += self._build_tree_string(node.sub_keywords, depth + 1)
        return tree_str


class RootKeywords(BaseModel):

    """Schema for representing root keywords and their hierarchies."""

    keyword_hierarchies: list[KeywordsHierarchy] | None = None
    """
    A list of root keywords with their sub-keywords.
    """

    model_config = ConfigDict(extra="forbid")


class KeywordsState(Keywords, RootKeywords):

    """State for extracting keywords from text."""

    input_text: str
    """
    The text to extract keywords from.
    """
