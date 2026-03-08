import logging

import uvicorn
from fastapi import FastAPI

from gargoyle.controller.mind_map_controller import router as mind_map_router
from gargoyle.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(
    router=mind_map_router,
    prefix="/mind-map",
    tags=["mind_map"],
)

if __name__ == "__main__":
    rest_api_settings = settings.rest_api
    uvicorn.run(
        app=app,
        host=rest_api_settings.host,
        port=rest_api_settings.port,
    )
