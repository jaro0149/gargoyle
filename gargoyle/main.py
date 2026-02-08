from gargoyle.graph.mind_map_config import MindMapConfig
from gargoyle.graph.mind_map_context import MindMapContext
from gargoyle.graph.mind_map_graph_builder import MindMapGraphBuilder
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState


def main():
    mind_map_graph_builder = MindMapGraphBuilder()
    graph = mind_map_graph_builder.build_mind_map_creation_graph()

    ospf_text = """
    Open Shortest Path First (OSPF) is a routing protocol for Internet Protocol (IP) networks.\
    It uses a link state routing (LSR) algorithm and falls into the group of interior gateway protocols (IGPs),\
    operating within a single autonomous system (AS).

    OSPF version 3 introduces modifications to the IPv4 implementation of the protocol.[2] Except for virtual links,\
    all neighbor exchanges use IPv6 link-local addressing exclusively. The IPv6 protocol runs per link, rather than\
    on the subnet. All IP prefix information has been removed from the link-state advertisements and from the hello\
    discovery packet, making OSPFv3 essentially protocol-independent. Despite the expanded IP addressing to 128 bits\
    in IPv6, area and router Identifications are still based on 32-bit numbers.\
    """

    context = MindMapContext(
        config=MindMapConfig()
    )
    res = graph.invoke(
        AggregatedKeywordsState(text=ospf_text),
        context=context
    )
    res_state = AggregatedKeywordsState.model_validate(res)
    print(res_state.mind_map_puml)


if __name__ == "__main__":
    main()
