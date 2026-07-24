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

active_tracking_task: asyncio.Task | None = None
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


# ----------------------------------------------------------------------
# Live Tracking
# ----------------------------------------------------------------------

@router.websocket("/track/live")
async def track_live(websocket: WebSocket):
    global active_tracking_task

    #
    # Kill any stale websocket loop (Gradio reconnect / proxy restart)
    #
    if (
        active_tracking_task is not None
        and not active_tracking_task.done()
    ):
        print("[WARN] Existing tracking loop detected. Cancelling...")

        active_tracking_task.cancel()

        try:
            await active_tracking_task
        except asyncio.CancelledError:
            pass

        active_tracking_task = None

    await websocket.accept()

    tracker = get_tracker()

    if not tracker.initialized:
        await websocket.send_json(
            {"error": "tracker not initialized"}
        )
        await websocket.close()
        return

    current_task = asyncio.current_task()
    active_tracking_task = current_task

    print("[INFO] Live tracking websocket connected.")

    try:
        while True:
            data = await websocket.receive_bytes()

            frame = decode_image(data)

            if frame is None:
                await websocket.send_json(
                    {"error": "decode_image returned None"}
                )
                continue

            #
            # Only one inference at a time
            #
            async with tracking_lock:
                result = tracker.track_step(frame)

            await websocket.send_json({
                "bbox": result["bbox"],
                "score": result["score"],
                "lost": result["lost"],
                "model_fps": result["model_fps"],
                "tracker_fps": result["tracker_fps"],
                "backend": result["backend"],
            })

    except WebSocketDisconnect as e:
        if e.code == 1012:
            print("[WARN] Client disconnected (1012 Service Restart)")
        else:
            print(f"[INFO] Client disconnected ({e.code})")

    except asyncio.CancelledError:
        print("[INFO] Tracking task cancelled.")
        raise

    except Exception as e:
        print(f"[ERROR] Live tracking failed: {e}")

    finally:
        if active_tracking_task is current_task:
            active_tracking_task = None

        print("[INFO] Tracking loop cleaned up.")


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