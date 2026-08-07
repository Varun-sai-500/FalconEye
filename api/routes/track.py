import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, status
import numpy as np
import cv2
import json
import tempfile
import shutil
from pathlib import Path
import av
from services.tracking_service import create_tracker

router = APIRouter()
executor = ThreadPoolExecutor(max_workers=1)


def decode_h264_packet(codec_context: av.CodecContext, packet_bytes: bytes) -> np.ndarray | None:
    try:
        packet = av.Packet(packet_bytes)
        frames = codec_context.decode(packet)
        for frame in frames:
            return frame.to_ndarray(format="bgr24")
    except Exception as e:
        print(f"[ERROR] H.264 Decoding Exception: {e}")
        return None
    return None

@router.websocket("/track/live")
async def track_live(websocket: WebSocket):
    await websocket.accept()

    loop = asyncio.get_running_loop()
    tracker = await loop.run_in_executor(executor, create_tracker)

    # Initialize low-latency H.264 decoder context
    codec_context = av.CodecContext.create('h264', 'r')
    codec_context.thread_type = 'NONE'
    codec_context.options = {'flags': 'low_delay'}
    codec_context.open()

    try:
        # Step 1: Receive Initial Bounding Box Payload
        init_message = await websocket.receive_text()
        try:
            init_data = json.loads(init_message)
            bbox = tuple(init_data["bbox"])
        except (json.JSONDecodeError, KeyError, TypeError):
            await websocket.send_json({"error": "Invalid init payload."})
            await websocket.close()
            return

        # Step 2: Receive and decode keyframe to initialize tracker
        first_frame = None
        while first_frame is None:
            first_packet_bytes = await websocket.receive_bytes()
            first_frame = await loop.run_in_executor(
                executor, decode_h264_packet, codec_context, first_packet_bytes
            )

        await loop.run_in_executor(executor, tracker.init_from_bbox, first_frame, bbox)
        await websocket.send_json({"status": "initialized", "bbox": list(bbox)})

        # Step 3: Tracking Loop
        while True:
            data = await websocket.receive_bytes()

            frame = await loop.run_in_executor(
                executor, decode_h264_packet, codec_context, data
            )

            if frame is None:
                continue

            result = await loop.run_in_executor(executor, tracker.tracking, frame)

            await websocket.send_json({
                "bbox": result["bbox"],
                "score": result["score"],
                "lost": result["lost"],
                "tracker_fps": result["tracker_fps"],
                "backend": result["backend"],
            })

    except WebSocketDisconnect:
        print("[INFO] Live tracking WebSocket disconnected safely.")
    finally:
        if hasattr(tracker, 'reset'):
            await loop.run_in_executor(executor, tracker.reset)

# ----------------------------------------------------------------------
# Unified Offline Tracking (HTTP) - Unchanged (Video Upload file remains MP4/HTTP)
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

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        shutil.copyfileobj(video.file, tmp)
        video_path = tmp.name
    loop = asyncio.get_running_loop()

    track_video = await loop.run_in_executor(executor, create_tracker)

    try:
        metrics = await loop.run_in_executor(
            executor,
            track_video.track_offline,
            video_path,
            bbox_parsed,
            False
        )
    finally:
        track_video.reset()
        path = Path(video_path)
        if path.exists():
            path.unlink()

    return {
        "status": "done",
        "metrics": metrics
    }