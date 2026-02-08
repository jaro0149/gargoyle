from langchain_core.language_models import BaseChatModel
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from gargoyle.edges.fan_out_keywords_extraction import fan_out_keywords_extraction
from gargoyle.edges.fan_out_keywords_merging import fan_out_merging_of_keywords
from gargoyle.edges.routing_after_keywords_extraction import route_after_keywords_extraction
from gargoyle.graph.node_identifiers import (
    ID_BUILD_KEYWORDS_HIERARCHIES,
    ID_CREATE_HIERARCHY,
    ID_EXTRACT_KEYWORDS,
    ID_JOIN_KEYWORDS_HIERARCHIES,
    ID_MERGE_HIERARCHIES,
    ID_PREPARE_KEYWORDS_BEFORE_MERGING, ID_BUILD_MIND_MAP, ID_SPLIT_TEXT,
)
from gargoyle.nodes.input_text_splitter import split_text
from gargoyle.nodes.keywords_extractor import KeywordsExtractor
from gargoyle.nodes.keywords_hierarchy_builder import KeywordsHierarchyBuilder
from gargoyle.nodes.merge_keyword_hierarchies import MergeKeywordHierarchies
from gargoyle.nodes.mind_map_builder import build_mind_map
from gargoyle.nodes.prepare_keywords_before_merging import prepare_keywords_before_merging
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState
from gargoyle.state.keywords_state import KeywordsState


class MindMapGraphBuilder:

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    def build_mind_map_creation_graph(self) -> CompiledStateGraph:
        key_extraction_graph = self._build_keywords_extraction_graph()
        return self._build_aggregation_graph(
            key_extraction_graph=key_extraction_graph
        )

    def _build_keywords_extraction_graph(self) -> CompiledStateGraph:
        extractor = KeywordsExtractor(model=self.llm)
        hierarchy_builder = KeywordsHierarchyBuilder(model=self.llm)

        graph_builder = StateGraph(KeywordsState)
        graph_builder.add_node(node=ID_EXTRACT_KEYWORDS, action=extractor)
        graph_builder.add_node(node=ID_CREATE_HIERARCHY, action=hierarchy_builder)

        graph_builder.add_edge(start_key=START, end_key=ID_EXTRACT_KEYWORDS)
        graph_builder.add_edge(start_key=ID_EXTRACT_KEYWORDS, end_key=ID_CREATE_HIERARCHY)
        graph_builder.add_edge(start_key=ID_CREATE_HIERARCHY, end_key=END)
        return graph_builder.compile()

    def _build_aggregation_graph(
            self,
            key_extraction_graph: CompiledStateGraph
    ) -> CompiledStateGraph:
        merge_hierarchies = MergeKeywordHierarchies(
            model=self.llm
        )

        graph_builder = StateGraph(AggregatedKeywordsState)
        graph_builder.add_node(node=ID_SPLIT_TEXT, action=split_text)
        graph_builder.add_node(node=ID_BUILD_KEYWORDS_HIERARCHIES, action=key_extraction_graph)
        graph_builder.add_node(node=ID_JOIN_KEYWORDS_HIERARCHIES, action=lambda state: state)
        graph_builder.add_node(node=ID_PREPARE_KEYWORDS_BEFORE_MERGING, action=prepare_keywords_before_merging)
        graph_builder.add_node(node=ID_MERGE_HIERARCHIES, action=merge_hierarchies)
        graph_builder.add_node(node=ID_BUILD_MIND_MAP, action=build_mind_map)

        graph_builder.add_edge(start_key=START, end_key=ID_SPLIT_TEXT)
        graph_builder.add_conditional_edges(
            source=ID_SPLIT_TEXT,
            path=fan_out_keywords_extraction,
            path_map=[ID_BUILD_KEYWORDS_HIERARCHIES, END]
        )
        graph_builder.add_edge(start_key=ID_BUILD_KEYWORDS_HIERARCHIES, end_key=ID_JOIN_KEYWORDS_HIERARCHIES)
        graph_builder.add_conditional_edges(
            source=ID_JOIN_KEYWORDS_HIERARCHIES,
            path=route_after_keywords_extraction,
            path_map=[ID_PREPARE_KEYWORDS_BEFORE_MERGING, ID_BUILD_MIND_MAP]
        )
        graph_builder.add_conditional_edges(
            source=ID_PREPARE_KEYWORDS_BEFORE_MERGING,
            path=fan_out_merging_of_keywords,
            path_map=[ID_MERGE_HIERARCHIES, ID_BUILD_MIND_MAP, END]
        )
        graph_builder.add_edge(start_key=ID_MERGE_HIERARCHIES, end_key=ID_PREPARE_KEYWORDS_BEFORE_MERGING)
        graph_builder.add_edge(start_key=ID_BUILD_MIND_MAP, end_key=END)
        return graph_builder.compile()
