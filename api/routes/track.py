from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, Form, File
from pydantic import BaseModel
import asyncio
import base64
import cv2
import tempfile
import shutil
import numpy as np

from services.tracking_service import (
    create_tracker,
    get_tracker,
    reset_tracker,
)

router = APIRouter()


# ----------------------------------------------------------------------
# Globals
# ----------------------------------------------------------------------


tracking_lock = asyncio.Lock()


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------

class InitTrackResponse(BaseModel):
    bbox: list[int]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def decode_image(file_bytes: bytes) -> np.ndarray:
    return cv2.imdecode(
        np.frombuffer(file_bytes, np.uint8),
        cv2.IMREAD_COLOR,
    )


def decode_mask_b64(mask_b64: str) -> np.ndarray:
    mask_bytes = base64.b64decode(mask_b64)
    return cv2.imdecode(
        np.frombuffer(mask_bytes, np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )


# ----------------------------------------------------------------------
# Init
# ----------------------------------------------------------------------

@router.post("/track/init", response_model=InitTrackResponse)
async def init_track(file: UploadFile, mask_b64: str = Form(...)):
    tracker = create_tracker()

    frame = decode_image(await file.read())
    mask = decode_mask_b64(mask_b64)

    bbox = tracker.init_from_mask(frame, mask)

    return InitTrackResponse(bbox=list(bbox))


connection_lock = asyncio.Lock()
active_tracking_task: asyncio.Task | None = None
generation = 0

@router.websocket("/track/live")
async def track_live(websocket: WebSocket):
    global active_tracking_task, generation

    async with connection_lock:
        if active_tracking_task is not None and not active_tracking_task.done():
            active_tracking_task.cancel()
            try:
                await active_tracking_task
            except asyncio.CancelledError:
                pass

        await websocket.accept()

        tracker = get_tracker()
        if not tracker.initialized:
            await websocket.send_json({"error": "tracker not initialized"})
            await websocket.close()
            return

        generation += 1
        my_gen = generation
        current_task = asyncio.current_task()
        active_tracking_task = current_task

    # ... loop body unchanged, but guard writes to shared state:
    try:
        while True:
            data = await websocket.receive_bytes()
            frame = decode_image(data)
            if frame is None:
                await websocket.send_json({"error": "decode_image returned None"})
                continue

            async with tracking_lock:
                if my_gen != generation:
                    # someone superseded us between recv and lock acquisition
                    break
                result = tracker.track_step(frame)

            await websocket.send_json({
                "bbox": result["bbox"],
                "score": result["score"],
                "lost": result["lost"],
                "model_fps": result["model_fps"],
                "tracker_fps": result["tracker_fps"],
                "backend": result["backend"],
            })
    finally:
        async with connection_lock:
            if active_tracking_task is current_task:
                active_tracking_task = None


@router.post("/track/debug")
async def debug(video: UploadFile = File(...)):
    tracker = get_tracker()

    with tempfile.NamedTemporaryFile(suffix=".mov", delete=False) as tmp:
        shutil.copyfileobj(video.file, tmp)
        video_path = tmp.name

    for _ in tracker.track_live(video_path, display=False):
        pass

    return {"status": "done"}

# ----------------------------------------------------------------------
# Reset
# ----------------------------------------------------------------------

@router.post("/track/reset")
def reset():
    reset_tracker()
    return {
        "status": "tracker reset"
    }