import asyncio
import logging

from gargoyle.graph.mind_map_config import MindMapConfig
from gargoyle.graph.mind_map_context import MindMapContext
from gargoyle.graph.mind_map_graph_builder import MindMapGraphBuilder
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState

logger = logging.getLogger(__name__)


async def _main() -> None:
    logging.basicConfig(level=logging.INFO)

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
        config=MindMapConfig(),
    )

    logger.info("Starting mind map creation")
    final_state = None
    async for _, event_type, event in graph.astream(
            AggregatedKeywordsState(text=ospf_text),
            context=context,
            stream_mode=["custom", "values"],
            subgraphs=True,
    ):
        if event_type == "custom":
            logger.info("Custom event: %s", event)
        elif event_type == "values":
            final_state = event

    if final_state:
        res_state = AggregatedKeywordsState.model_validate(final_state)
        logger.info("Resulting PlantUML:")
        logger.info(res_state.mind_map_puml)
    else:
        logger.error("No final state found!")


if __name__ == "__main__":
    asyncio.run(_main())
