import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, status
import numpy as np
import cv2
import json
import tempfile
import shutil
from pathlib import Path

router = APIRouter()

# Create a dedicated thread pool for heavy CPU/GPU processing
# This keeps the main async event loop completely free to handle network packets
executor = ThreadPoolExecutor(max_workers=1)

from services.tracking_service import create_tracker

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def decode_image(file_bytes: bytes) -> np.ndarray:
    return cv2.imdecode(
        np.frombuffer(file_bytes, np.uint8),
        cv2.IMREAD_COLOR,
    )

# ----------------------------------------------------------------------
# Unified Live Tracking (WebSocket)
# ----------------------------------------------------------------------

@router.websocket("/track/live")
async def track_live(websocket: WebSocket):
    await websocket.accept()

    # Instantiate a clean, isolated tracker session dedicated to this connection lifetime
    loop = asyncio.get_running_loop()
    tracker = await loop.run_in_executor(executor, create_tracker)

    try:
        # Step 1: Initialize Bounding Box Payload
        init_message = await websocket.receive_text()
        try:
            init_data = json.loads(init_message)
            bbox = tuple(init_data["bbox"])
        except (json.JSONDecodeError, KeyError, TypeError):
            await websocket.send_json({"error": "Invalid init payload."})
            await websocket.close()
            return

        # Step 2: Await the first raw frame bytes
        first_frame_bytes = await websocket.receive_bytes()

        # Offload image decoding to the thread pool
        first_frame = await loop.run_in_executor(executor, decode_image, first_frame_bytes)
        if first_frame is None:
            await websocket.send_json({"error": "Failed to decode initial frame"})
            await websocket.close()
            return

        # Offload model template initialization to the thread pool
        await loop.run_in_executor(executor, tracker.init_from_bbox, first_frame, bbox)
        await websocket.send_json({"status": "initialized", "bbox": list(bbox)})

        # Step 3: High-Frequency Inference Streaming Loop
        while True:
            data = await websocket.receive_bytes()

            # Offload decoding to keep the socket read buffer clear
            frame = await loop.run_in_executor(executor, decode_image, data)
            if frame is None:
                await websocket.send_json({"error": "decode_image returned None"})
                continue

            # Offload the core neural net pass to the thread pool
            result = await loop.run_in_executor(executor, tracker.track_step, frame)

            # Send structured primitives back to client
            await websocket.send_json({
                "bbox": result["bbox"],
                "score": result["score"],
                "lost": result["lost"],
                "tracker_fps": result["tracker_fps"],
                "backend": result["backend"],
            })

    except WebSocketDisconnect:
        print("[INFO] Live tracking WebSocket disconnected client safely.")
    finally:
        # Explicit clean up step if your wrapper manages pinned CUDA buffers
        if hasattr(tracker, 'reset'):
            await loop.run_in_executor(executor, tracker.reset)
            
# ----------------------------------------------------------------------
# Unified Offline Tracking (HTTP)
# ----------------------------------------------------------------------
@router.post("/track_video")
async def track_video(
    video: UploadFile = File(...),
    bbox: str = Form(...)  # Expected JSON string: "[x, y, w, h]"
):
    try:
        bbox_parsed = tuple(json.loads(bbox))
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid bbox format. Use JSON array '[x, y, w, h]'."
        )

    # Write file out safely using a contextual block
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        shutil.copyfileobj(video.file, tmp)
        video_path = tmp.name

    loop = asyncio.get_running_loop()

    # Instantiating a distinct offline tracker instance prevents cross-talk with live sockets
    track_video = await loop.run_in_executor(executor, create_tracker)

    try:
        # Offload the entire blocking file profiling sweep to the thread pool
        metrics = await loop.run_in_executor(
            executor,
            track_video.track_live,
            video_path,
            bbox_parsed,
            False
        )
    finally:
        track_video.reset()
        # Clean up temporary disk storage footprint
        path = Path(video_path)
        if path.exists():
            path.unlink()

    return {
        "status": "done",
        "metrics": metrics
    }