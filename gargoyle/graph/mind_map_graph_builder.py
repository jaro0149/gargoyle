from typing import Any, cast

from langgraph.constants import END, START
from langgraph.graph import StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph.state import CompiledStateGraph  # pyright: ignore[reportMissingTypeStubs]

from gargoyle.edges.fan_out_keywords_extraction import fan_out_keywords_extraction
from gargoyle.edges.fan_out_keywords_merging import fan_out_merging_of_keywords
from gargoyle.edges.route_keywords_extraction import route_keywords_extraction
from gargoyle.edges.routing_after_keywords_extraction import route_after_keywords_extraction
from gargoyle.graph.mind_map_context import MindMapContext
from gargoyle.graph.node_identifiers import (
    ID_BUILD_KEYWORDS_HIERARCHIES,
    ID_BUILD_MIND_MAP,
    ID_CREATE_HIERARCHY,
    ID_EXTRACT_KEYWORDS,
    ID_EXTRACT_KEYWORDS_SINGLE_STEP,
    ID_JOIN_KEYWORDS_HIERARCHIES,
    ID_MERGE_HIERARCHIES,
    ID_PREPARE_KEYWORDS_BEFORE_MERGING,
    ID_SPLIT_TEXT,
)
from gargoyle.llm.llm_factory import LLMFactory
from gargoyle.nodes.input_text_splitter import split_text
from gargoyle.nodes.keywords_extractor import KeywordsExtractor
from gargoyle.nodes.keywords_hierarchy_builder import KeywordsHierarchyBuilder
from gargoyle.nodes.keywords_single_step_builder import KeywordsSingleStepBuilder
from gargoyle.nodes.merge_keyword_hierarchies import MergeKeywordHierarchies
from gargoyle.nodes.mind_map_builder import build_mind_map
from gargoyle.nodes.prepare_keywords_before_merging import prepare_keywords_before_merging
from gargoyle.settings import settings
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState
from gargoyle.state.keywords_state import KeywordsState


class MindMapGraphBuilder:

    """
    A builder for constructing a mind map creation process using state graphs.

    This class provides functionality to construct and compile modular state graphs that handle
    keyword extraction and aggregation, ultimately producing a mind map creation flow. The APIs
    allow for the creation of reusable components using language model-based processing, making
    it extensible for various text-processing tasks.
    """

    def __init__(self) -> None:
        """Initialize an instance of the class."""
        self.llm_factory = LLMFactory()

    def build_mind_map_creation_graph(
            self,
    ) -> CompiledStateGraph[AggregatedKeywordsState, MindMapContext, AggregatedKeywordsState, AggregatedKeywordsState]:
        """
        Build and return a compiled state graph for mind map creation by combining keyword extraction and aggregation.

        The method creates a modular structure where keywords are first extracted and then aggregated.

        :return: Compiled state graph representing the mind map creation process.
        """
        key_extraction_graph = self._build_keywords_extraction_graph()
        return self._build_aggregation_graph(
            key_extraction_graph=key_extraction_graph,
        )


    def _build_keywords_extraction_graph(
            self,
    ) -> CompiledStateGraph[KeywordsState, MindMapContext, KeywordsState, KeywordsState]:
        extractor = KeywordsExtractor(
            model=self.llm_factory.get_llm(settings.graph_nodes.keywords_extractor_id),
        )
        hierarchy_builder = KeywordsHierarchyBuilder(
            model=self.llm_factory.get_llm(settings.graph_nodes.keywords_hierarchy_builder_id),
        )
        single_step_builder = KeywordsSingleStepBuilder(
            model=self.llm_factory.get_llm(settings.graph_nodes.keywords_single_step_builder_id),
        )

        graph_builder = StateGraph(KeywordsState, context_schema=MindMapContext)
        # Routed add_node(...) and compile() calls through a local Any-typed builder handle to avoid
        # reportUnknownMemberType noise from langgraph partial typing
        graph_builder_api: Any = graph_builder
        graph_builder_api.add_node(node=ID_EXTRACT_KEYWORDS, action=extractor)
        graph_builder_api.add_node(node=ID_CREATE_HIERARCHY, action=hierarchy_builder)
        graph_builder_api.add_node(node=ID_EXTRACT_KEYWORDS_SINGLE_STEP, action=single_step_builder)

        graph_builder.add_conditional_edges(
            source=START,
            path=route_keywords_extraction,
            path_map=[ID_EXTRACT_KEYWORDS, ID_EXTRACT_KEYWORDS_SINGLE_STEP],
        )
        graph_builder.add_edge(start_key=ID_EXTRACT_KEYWORDS, end_key=ID_CREATE_HIERARCHY)
        graph_builder.add_edge(start_key=ID_CREATE_HIERARCHY, end_key=END)
        graph_builder.add_edge(start_key=ID_EXTRACT_KEYWORDS_SINGLE_STEP, end_key=END)
        return cast(
            "CompiledStateGraph[KeywordsState, MindMapContext, KeywordsState, KeywordsState]",
            graph_builder_api.compile(),
        )

    def _build_aggregation_graph(
            self,
            key_extraction_graph: CompiledStateGraph[KeywordsState, MindMapContext, KeywordsState, KeywordsState],
    ) -> CompiledStateGraph[AggregatedKeywordsState, MindMapContext, AggregatedKeywordsState, AggregatedKeywordsState]:
        merge_hierarchies = MergeKeywordHierarchies(
            model=self.llm_factory.get_llm(settings.graph_nodes.merge_keyword_hierarchies_id),
        )
        def join_keywords_hierarchies(_: AggregatedKeywordsState) -> dict[str, object]:
            return {}

        graph_builder = StateGraph(AggregatedKeywordsState, context_schema=MindMapContext)
        # Routed add_node(...) and compile() calls through a local Any-typed builder handle to avoid
        # reportUnknownMemberType noise from langgraph partial typing
        graph_builder_api: Any = graph_builder
        graph_builder_api.add_node(node=ID_SPLIT_TEXT, action=split_text)
        graph_builder_api.add_node(node=ID_BUILD_KEYWORDS_HIERARCHIES, action=key_extraction_graph)
        graph_builder_api.add_node(node=ID_JOIN_KEYWORDS_HIERARCHIES, action=join_keywords_hierarchies)
        graph_builder_api.add_node(node=ID_PREPARE_KEYWORDS_BEFORE_MERGING, action=prepare_keywords_before_merging)
        graph_builder_api.add_node(node=ID_MERGE_HIERARCHIES, action=merge_hierarchies)
        graph_builder_api.add_node(node=ID_BUILD_MIND_MAP, action=build_mind_map)

        graph_builder.add_edge(start_key=START, end_key=ID_SPLIT_TEXT)
        graph_builder.add_conditional_edges(
            source=ID_SPLIT_TEXT,
            path=fan_out_keywords_extraction,
            path_map=[ID_BUILD_KEYWORDS_HIERARCHIES, END],
        )
        graph_builder.add_edge(start_key=ID_BUILD_KEYWORDS_HIERARCHIES, end_key=ID_JOIN_KEYWORDS_HIERARCHIES)
        graph_builder.add_conditional_edges(
            source=ID_JOIN_KEYWORDS_HIERARCHIES,
            path=route_after_keywords_extraction,
            path_map=[ID_PREPARE_KEYWORDS_BEFORE_MERGING, ID_BUILD_MIND_MAP],
        )
        graph_builder.add_conditional_edges(
            source=ID_PREPARE_KEYWORDS_BEFORE_MERGING,
            path=fan_out_merging_of_keywords,
            path_map=[ID_MERGE_HIERARCHIES, ID_BUILD_MIND_MAP, END],
        )
        graph_builder.add_edge(start_key=ID_MERGE_HIERARCHIES, end_key=ID_PREPARE_KEYWORDS_BEFORE_MERGING)
        graph_builder.add_edge(start_key=ID_BUILD_MIND_MAP, end_key=END)
        return cast(
            "CompiledStateGraph[AggregatedKeywordsState, MindMapContext, AggregatedKeywordsState, AggregatedKeywordsState]", # noqa: E501
            graph_builder_api.compile(),
        )
