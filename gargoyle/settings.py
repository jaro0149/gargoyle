from pydantic.v1 import BaseSettings


class KeywordExtractorSettings(BaseSettings):
    """
    Settings for the keyword extractor node.
    """
    max_keywords: int = 10
    """
    Maximum number of keywords to extract from the input text.
    """
    max_words_in_keyword: int = 3
    """
    Maximum number of words in a single keyword.
    """


class KeywordsHierarchySettings(BaseSettings):
    """
    Settings for the keywords hierarchy builder node.
    """
    max_depth: int = 3
    """
    Maximum depth of the keywords tree hierarchy (including the root keyword).
    """


class KeywordsMergingSetting(BaseSettings):
    """
    Settings for the keywords merging node.
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


class Settings(BaseSettings):
    """
    Global settings for the Gargoyle application.
    """
    keywords_extractor: KeywordExtractorSettings = KeywordExtractorSettings()
    """
    Settings for the keyword extractor node.
    """
    keywords_hierarchy: KeywordsHierarchySettings = KeywordsHierarchySettings()
    """
    Settings for the keywords hierarchy builder node.
    """
    merge_keywords: KeywordsMergingSetting = KeywordsMergingSetting()
    """
    Settings for the keywords merging node.
    """
