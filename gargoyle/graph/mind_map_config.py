from pydantic import BaseModel


class KeywordsExtractorConfig(BaseModel):
    """
    Configuration for the keyword extractor node.
    """
    max_keywords: int = 10
    """
    Maximum number of keywords to extract from the input text.
    """
    max_words_in_keyword: int = 3
    """
    Maximum number of words in a single keyword.
    """


class KeywordsHierarchyConfig(BaseModel):
    """
    Configuration for the keywords hierarchy builder node.
    """
    max_depth: int = 3
    """
    Maximum depth of the keywords tree hierarchy (including the root keyword).
    """


class KeywordsMergingConfig(BaseModel):
    """
    Configuration for the keywords merging node.
    """
    enabled: bool = True
    """
    Whether to enable the keywords merging node.
    If disabled, the output of the keywords hierarchy builder node is returned directly.
    """
    randomize_keywords: bool = True
    """
    Whether to randomize the order of root keywords before merging trees.
    """
    max_root_keywords: int = 2
    """
    Maximum number of root keywords in the final merged structure.
    If the number of root keywords exceeds this value, the root hierarchies are squashed into smaller number.
    """
    squash_root_keywords: int = 30
    """
    Number of root keywords to squash into up to max_root_keywords.
    """


class Config(BaseModel):
    """
    Global configuration for the Gargoyle application.
    """
    keywords_extractor: KeywordsExtractorConfig = KeywordsExtractorConfig()
    """
    Configuration for the keyword extractor node.
    """
    keywords_hierarchy: KeywordsHierarchyConfig = KeywordsHierarchyConfig()
    """
    Configuration for the keywords hierarchy builder node.
    """
    merge_keywords: KeywordsMergingConfig = KeywordsMergingConfig()
    """
    Configuration for the keywords merging node.
    """
