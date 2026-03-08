from pydantic import BaseModel, Field

from gargoyle.graph.mind_map_config import MindMapConfig


class GenerateRequest(BaseModel):

    """Request model for the /generate mind-map endpoint."""

    text: str = Field(
        description="The input text to process for mind map generation.",
    )
    config: MindMapConfig = Field(
        default_factory=MindMapConfig,
        description="Configuration for the mind map generation.",
    )
