from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import SecretStr

from gargoyle.edges.fan_out_keywords_extraction import fan_out_keywords_extraction
from gargoyle.edges.fan_out_keywords_merging import FanOutKeywordsMerging
from gargoyle.nodes.keywords_extractor import KeywordsExtractor
from gargoyle.nodes.keywords_hierarchy_builder import KeywordsHierarchyBuilder
from gargoyle.nodes.merge_keyword_hierarchies import MergeKeywordHierarchies
from gargoyle.nodes.prepare_keywords_before_merging import PrepareKeywordsBeforeMerging
from gargoyle.settings import Settings
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState
from gargoyle.state.keywords_state import KeywordsState


def main() -> None:
    llm = ChatOpenAI(
        model="gpt-5-nano",
        api_key=None,
        streaming=True,
        # max_tokens=1024
    )

    settings = Settings()
    extractor = KeywordsExtractor(model=llm, settings=settings.keywords_extractor)
    hierarchy_builder = KeywordsHierarchyBuilder(model=llm, settings=settings.keywords_hierarchy)
    merge_hierarchies = MergeKeywordHierarchies(
        model=llm,
        hierarchy_settings=settings.keywords_hierarchy,
        merge_settings=settings.merge_keywords
    )
    prepare_keywords_before_merging = PrepareKeywordsBeforeMerging(settings.merge_keywords)
    fan_out_keywords_merging = FanOutKeywordsMerging(settings=settings.merge_keywords)

    # Subgraph for processing a single text
    subgraph_builder = StateGraph(KeywordsState)
    subgraph_builder.add_node(node="extract_keywords", action=extractor)
    subgraph_builder.add_node(node="create_hierarchy", action=hierarchy_builder)
    subgraph_builder.add_edge(start_key=START, end_key="extract_keywords")
    subgraph_builder.add_edge(start_key="extract_keywords", end_key="create_hierarchy")
    subgraph_builder.add_edge(start_key="create_hierarchy", end_key=END)
    subgraph = subgraph_builder.compile()

    # Parent graph
    graph_builder = StateGraph(AggregatedKeywordsState)
    graph_builder.add_node(node="build_keywords_hierarchies", action=subgraph)
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
    graph = graph_builder.compile()

    ospf_text_1 = """
    Open Shortest Path First (OSPF) is a routing protocol for Internet Protocol (IP) networks. 
    It uses a link state routing (LSR) algorithm and falls into the group of interior gateway protocols (IGPs), 
    operating within a single autonomous system (AS).
    """

    ospf_text_2 = """
    OSPF version 3 introduces modifications to the IPv4 implementation of the protocol.[2] Except for virtual links, 
    all neighbor exchanges use IPv6 link-local addressing exclusively. The IPv6 protocol runs per link, rather than based 
    on the subnet. All IP prefix information has been removed from the link-state advertisements and from the hello 
    discovery packet, making OSPFv3 essentially protocol-independent. Despite the expanded IP addressing to 128 bits
    in IPv6, area and router Identifications are still based on 32-bit numbers. 
    """

    res = graph.invoke(AggregatedKeywordsState(input_texts=[ospf_text_1, ospf_text_2]))
    print(res)


if __name__ == "__main__":
    main()
