from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from gargoyle.graph.mind_map_config import Config
from gargoyle.graph.mind_map_graph_builder import build_mind_map_creation_graph
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState


def main():
    llm = ChatOpenAI(
        model="gpt-5-nano",
        api_key=None,
        streaming=True,
        # max_tokens=1024
    )

    config = Config()
    graph = build_mind_map_creation_graph(
        config=config,
        llm=llm
    )

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
