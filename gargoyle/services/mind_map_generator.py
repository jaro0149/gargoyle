import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from langgraph.graph.state import CompiledStateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.types import Overwrite
from pydantic import BaseModel

from gargoyle.graph.mind_map_config import MindMapConfig
from gargoyle.graph.mind_map_context import MindMapContext
from gargoyle.graph.mind_map_graph_builder import MindMapGraphBuilder
from gargoyle.state.aggregated_keywords_state import AggregatedKeywordsState

logger = logging.getLogger(__name__)


class _MindMapEventsEncoder(json.JSONEncoder):

    """Custom JSON encoder to handle Pydantic models and Overwrite objects."""

    def default(self, o: Any) -> Any:  # noqa: ANN401
        if isinstance(o, BaseModel):
            return o.model_dump()
        if isinstance(o, Overwrite):
            return o.value
        return super().default(o)


class MindMapGenerator:

    """Service class responsible for generating mind maps from text input using a graph-based processing approach."""

    def __init__(self) -> None:
        """Initialize the MindMapGenerator by creating the mind map processing graph."""
        self.mind_map_graph = self._create_mind_map_graph()

    @staticmethod
    def _create_mind_map_graph(
    ) -> CompiledStateGraph[AggregatedKeywordsState, MindMapContext, AggregatedKeywordsState, AggregatedKeywordsState]:
        mind_map_graph_builder = MindMapGraphBuilder()
        return mind_map_graph_builder.build_mind_map_creation_graph()

    async def generate_mind_map(self, text: str, config: MindMapConfig) -> AsyncGenerator[str]:
        """
        Generate a mind map from the given text and configuration.

        Types of the output events:

        - Intermediate Events: `{"type": "custom", "event": {...}}` - These represent progress from different nodes
          in the processing graph.
        - Final State Event: `{"type": "values", "event": { "mind_map_puml": "...", ... }}` - The last successful event.
        - Error Event: `{"type": "error", "message": "..."}` - If an error occurs during processing.

        :param text: The input text to process for mind map generation.
        :param config: Configuration settings for the mind map generation process, including text splitting,
            keyword extraction, hierarchy building, and merging settings.
        :return: An asynchronous generator that yields JSON strings representing the events of the mind map gen process.
        """
        context = MindMapContext(config=config)
        logger.info("Starting mind map creation via API")
        last_values_event = None
        try:
            async for _, event_type, event in self.mind_map_graph.astream(  # pyright: ignore[reportUnknownMemberType]
                    input=AggregatedKeywordsState(text=text),
                    context=context,
                    stream_mode=["custom", "values"],
                    subgraphs=True,
            ):
                if event_type == "values":
                    last_values_event = event
                else:
                    yield json.dumps({"type": event_type, "event": event}, cls=_MindMapEventsEncoder) + "\n"

            if last_values_event is not None:
                yield json.dumps({"type": "values", "event": last_values_event}, cls=_MindMapEventsEncoder) + "\n"
        except Exception as e:
            logger.exception("Error during mind map generation")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
