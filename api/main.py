from fastapi import FastAPI
from api.routes import segment, track, follow
from services.segmentation_service import sam_service, clipseg_service
from services.tracking_service import get_tracker

app = FastAPI()
app.include_router(segment.router)
app.include_router(track.router)
app.include_router(follow.router)

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "sam": sam_service is not None,
        "clipseg": clipseg_service is not None,
        "tracker_initialized": get_tracker().state is not None,
    }