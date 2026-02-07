from langchain_core.language_models import BaseChatModel
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from gargoyle.edges.fan_out_keywords_extraction import fan_out_keywords_extraction
from gargoyle.edges.fan_out_keywords_merging import FanOutKeywordsMerging
from gargoyle.graph.mind_map_config import Config
from gargoyle.nodes.keywords_extractor import KeywordsExtractor
from gargoyle.nodes.keywords_hierarchy_builder import KeywordsHierarchyBuilder
from gargoyle.nodes.merge_keyword_hierarchies import MergeKeywordHierarchies
from gargoyle.nodes.prepare_keywords_before_merging import PrepareKeywordsBeforeMerging
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState
from gargoyle.state.keywords_state import KeywordsState


def build_mind_map_creation_graph(config: Config, llm: BaseChatModel) -> CompiledStateGraph:
    key_extraction_graph = _build_keywords_extraction_graph(
        config=config,
        llm=llm
    )
    return _build_aggregation_graph(
        config=config,
        llm=llm,
        key_extraction_graph=key_extraction_graph
    )


def _build_keywords_extraction_graph(config: Config, llm: BaseChatModel) -> CompiledStateGraph:
    extractor = KeywordsExtractor(model=llm, config=config.keywords_extractor)
    hierarchy_builder = KeywordsHierarchyBuilder(model=llm, config=config.keywords_hierarchy)

    graph_builder = StateGraph(KeywordsState)
    graph_builder.add_node(node="extract_keywords", action=extractor)
    graph_builder.add_node(node="create_hierarchy", action=hierarchy_builder)

    graph_builder.add_edge(start_key=START, end_key="extract_keywords")
    graph_builder.add_edge(start_key="extract_keywords", end_key="create_hierarchy")
    graph_builder.add_edge(start_key="create_hierarchy", end_key=END)
    return graph_builder.compile()


def _build_aggregation_graph(
        config: Config,
        llm: BaseChatModel,
        key_extraction_graph: CompiledStateGraph
) -> CompiledStateGraph:
    merge_hierarchies = MergeKeywordHierarchies(
        model=llm,
        hierarchy_config=config.keywords_hierarchy,
        merge_config=config.merge_keywords
    )
    prepare_keywords_before_merging = PrepareKeywordsBeforeMerging(config=config.merge_keywords)
    fan_out_keywords_merging = FanOutKeywordsMerging(config=config.merge_keywords)

    graph_builder = StateGraph(AggregatedKeywordsState)
    graph_builder.add_node(node="build_keywords_hierarchies", action=key_extraction_graph)
    graph_builder.add_node(node="prepare_keywords_before_merging", action=prepare_keywords_before_merging)
    graph_builder.add_node(node="merge_hierarchies", action=merge_hierarchies)

    graph_builder.add_conditional_edges(
        source=START,
        path=fan_out_keywords_extraction,
        path_map=["build_keywords_hierarchies", END]
    )
    graph_builder.add_edge(start_key="build_keywords_hierarchies", end_key="prepare_keywords_before_merging")
    graph_builder.add_conditional_edges(
        source="prepare_keywords_before_merging",
        path=fan_out_keywords_merging,
        path_map=["merge_hierarchies", END]
    )
    graph_builder.add_edge(start_key="merge_hierarchies", end_key="prepare_keywords_before_merging")
    return graph_builder.compile()
