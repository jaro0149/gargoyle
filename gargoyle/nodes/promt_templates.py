KEYWORDS_EXTRACTION_PROMPT = """\
You are an expert in extracting keywords from text.\
Limit output to {max_keywords} keywords.\
One keywork may contain up to {max_words_in_keyword} words.\
"""

KEYWORDS_HIERARCHY_CREATION_PROMPT = """\
You are an expert in organizing keywords into hierarchical structures.\
Create a hierarchy with up to {max_depth} levels based on the relationships between keywords.\
"""

COMBINE_HIERARCHIES_PROMPT = """\
You are an expert in merging hierarchical keyword structures.\
Combine the input groups of hierarchies into a single group of root hierarchies, eliminating duplicates,\
joining branches, and preserving relationships between keywords.\
Limit the number of root hierarchies to {max_root_keywords}.\
Limit the depth of the resulting hierarchies to {max_depth}.\
"""

KEYWORDS_SINGLE_STEP_PROMPT = """\
You are an expert in extracting keywords from text and organizing them into hierarchical structures.
Extract the most relevant keywords from the provided text and arrange them into a hierarchy.
Limit the total number of keywords to {max_keywords}.
Each keyword may contain up to {max_words_in_keyword} words.
Create a hierarchy with up to {max_depth} levels based on the relationships between keywords.
"""
