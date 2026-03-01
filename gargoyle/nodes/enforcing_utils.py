from gargoyle.graph.mind_map_config import KeywordsExtractorConfig, KeywordsHierarchyConfig
from gargoyle.state.keywords_state import Keywords, KeywordsHierarchy


def enforce_max_depth(config: KeywordsHierarchyConfig, hierarchy: KeywordsHierarchy) -> KeywordsHierarchy:
    """
    Enforces the maximum depth restriction on the given hierarchy based on the configuration.

    This function ensures that the hierarchy does not exceed the specified maximum
    depth provided in the config. It processes the hierarchy recursively and trims
    it to comply with the defined depth limitation.

    :param config: Configuration object defining the maximum depth.
    :param hierarchy: The hierarchy to be processed and restricted by the maximum depth.
    :return: Hierarchy modified to conform to the maximum depth constraint.
    """
    return _enforce_max_depth(max_depth=config.max_depth, hierarchy=hierarchy, current_depth=1)


def trim_keywords(config: KeywordsExtractorConfig, derived_keywords: Keywords) -> Keywords:
    """
    Trims the derived keywords based on the config, limiting the number of keywords and the length of each keyword.

    :param config: Configuration object defining the maximum keywords and words per keyword.
    :param derived_keywords: The keywords to be trimmed.
    :return: Keywords with trimmed content according to the configuration.
    """
    if not derived_keywords.keywords:
        return derived_keywords

    trimmed_keywords: list[str] = []
    for kw in derived_keywords.keywords[:config.max_keywords]:
        words = kw.split()
        trimmed_kw = " ".join(words[:config.max_words_in_keyword])
        trimmed_keywords.append(trimmed_kw)

    return Keywords(keywords=trimmed_keywords)


def _enforce_max_depth(max_depth: int, hierarchy: KeywordsHierarchy, current_depth: int) -> KeywordsHierarchy:
    if current_depth >= max_depth:
        return KeywordsHierarchy(keyword=hierarchy.keyword, sub_keywords=None)

    new_sub_keywords = None
    if hierarchy.sub_keywords:
        new_sub_keywords = [
            _enforce_max_depth(max_depth, sub_keyword, current_depth + 1)
            for sub_keyword in hierarchy.sub_keywords
        ]

    return KeywordsHierarchy(keyword=hierarchy.keyword, sub_keywords=new_sub_keywords)
