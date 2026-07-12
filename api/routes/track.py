from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, Form
from pydantic import BaseModel
import base64
import cv2
import numpy as np

from services.tracking_service import (
    create_tracker,
    get_tracker,
    reset_tracker,
)

router = APIRouter()


class InitTrackResponse(BaseModel):
    bbox: list[int]


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


@router.post("/track/init", response_model=InitTrackResponse)
async def init_track(file: UploadFile, mask_b64: str = Form(...)):
    tracker = create_tracker()

    frame = decode_image(await file.read())
    mask = decode_mask_b64(mask_b64)

    bbox = tracker.init_from_mask(frame, mask)

    return InitTrackResponse(bbox=list(bbox))
@router.websocket("/track/live")
async def track_live(websocket: WebSocket):
    await websocket.accept()

    tracker = get_tracker()

    if not tracker.initialized:
        await websocket.send_json(
            {"error": "tracker not initialized"}
        )
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_bytes()

            frame = decode_image(data)

            if frame is None:
                await websocket.send_json(
                    {"error": "decode_image returned None"}
                )
                continue

            result = tracker.track_step(frame)

            await websocket.send_json({
                "bbox": result["bbox"],
                "score": result["score"],
                "lost": result["lost"],
                "fps": result["fps"],
            })

    except WebSocketDisconnect:
        pass
    
@router.post("/track/reset")
def reset():
    reset_tracker()
    return {"status": "tracker reset"}