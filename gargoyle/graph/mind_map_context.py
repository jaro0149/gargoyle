from pydantic import BaseModel

from gargoyle.graph.mind_map_config import MindMapConfig


class MindMapContext(BaseModel):
    """
    Mind map graph context containing configuration and other dependencies.
    """
    config: MindMapConfig
    """
    Configuration used for generation of the mind map.
    """
