from fastapi import APIRouter
from starlette.responses import StreamingResponse

from gargoyle.controller.model import GenerateRequest
from gargoyle.services.mind_map_generator import MindMapGenerator

router = APIRouter()
mind_map_generator = MindMapGenerator()


@router.post("/generate")
async def generate_mind_map(request_data: GenerateRequest) -> StreamingResponse:
    """
    Endpoint to generate a mind map from text and configuration.

    The response is a stream of NDJSON (Newline Delimited JSON) events.

    Example request::

      {
        "text": "Artificial Intelligence is transforming industries. Machine learning is a subset of AI...",
        "config": {
          "text_splitter": { "chunk_size": 1000, "chunk_overlap": 100 },
          "keywords_hierarchy": { "use_single_step": true, "max_depth": 3 }
        }
      }

    Example output intermediate event::

      {
        "type": "custom",
        "event": {
          "node": "split_text",
          "status": "completed"
        }
      }

    Example output error event::

      {
        "type": "error",
        "message": "Failed to connect to LLM provider"
      }

    Example output final event::

      {
        "type": "values",
        "event": {
          "text": "...",
          "mind_map_puml": "...",
          "keyword_hierarchies": [...]
        }
      }

    :param request_data: The request data containing text and configuration for mind map generation.
    :return: A streaming response with NDJSON events representing the mind map generation process.
    """
    return StreamingResponse(
        mind_map_generator.generate_mind_map(request_data.text, request_data.config),
        media_type="application/x-ndjson",
    )
