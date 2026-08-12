import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, status
import json
import tempfile
import shutil
from pathlib import Path
import av
from services.tracking_service import create_tracker

router = APIRouter()

# Isolated thread pools
io_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="file_io")
decoder_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h264_decode")  # Serialized for thread-safety
tracker_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trt_tracker")  # Serialized CUDA/TRT context

def create_low_latency_codec_context() -> av.CodecContext:
    ctx = av.CodecContext.create("h264", "r")
    ctx.thread_type = "NONE"
    ctx.flags |= av.codec.context.Flags.low_delay
    ctx.options = {
        "fflags": "nobuffer",
    }
    ctx.open()
    return ctx

def decode_h264_packet_zero_copy(codec_context: av.CodecContext, packet_bytes: bytes):
    """Decodes H.264 packet without byte duplication using memoryview."""
    try:
        packet = av.Packet(memoryview(packet_bytes))
        frames = codec_context.decode(packet)
        for frame in frames:
            return frame.to_ndarray(format="bgr24")
    except Exception as e:
        print(f"[ERROR] H.264 Decoding Exception: {e}")
    return None

@router.websocket("/track/live")
async def track_live(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()

    # Reuse global single-user TensorRT tracker engine
    tracker = await loop.run_in_executor(tracker_executor, create_tracker)

    # Per-request, isolated decoder context
    codec_context = create_low_latency_codec_context()

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

        # Step 2: Receive and decode initial keyframe (with attempt threshold)
        first_frame = None
        attempts = 0
        max_attempts = 30  # Safety threshold against infinite loops on corrupted streams

        while first_frame is None:
            if attempts >= max_attempts:
                await websocket.send_json({"error": "Failed to decode initial keyframe."})
                await websocket.close()
                return

            first_packet_bytes = await websocket.receive_bytes()
            first_frame = await loop.run_in_executor(
                decoder_executor, 
                decode_h264_packet_zero_copy, 
                codec_context, 
                first_packet_bytes
            )
            attempts += 1

        await loop.run_in_executor(tracker_executor, tracker.init_from_bbox, first_frame, bbox)
        await websocket.send_json({"status": "initialized", "bbox": list(bbox)})

        # Step 3: Single-pass Live Tracking Loop
        while True:
            data = await websocket.receive_bytes()

            frame = await loop.run_in_executor(
                decoder_executor,
                decode_h264_packet_zero_copy,
                codec_context,
                data
            )

            if frame is None:
                continue

            result = await loop.run_in_executor(
                tracker_executor,
                tracker.tracking,
                frame
            )

            if result is None:
                continue        

            await websocket.send_json({
                "bbox": result["bbox"],
                "score": result["score"],
                "lost": result["lost"],
                "tracker_fps": result["tracker_fps"],
                "backend": result["backend"],
            })

    except WebSocketDisconnect:
        print("[INFO] Live tracking WebSocket disconnected safely.")


# ----------------------------------------------------------------------
# Async File I/O for Offline Video Tracking (HTTP POST)
# ----------------------------------------------------------------------
async def _write_upload_to_tempfile(upload_file: UploadFile) -> str:
    """Synchronous disk write offloaded to io_executor."""
    def _write_temp():
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            return tmp.name

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(io_executor, _write_temp)

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

    # Write video to disk in dedicated I/O pool
    video_path = await _write_upload_to_tempfile(video)
    loop = asyncio.get_running_loop()

    # Reuse global single-user TensorRT tracker engine
    tracker = await loop.run_in_executor(tracker_executor, create_tracker)
    metrics = None
    try:
        metrics = await loop.run_in_executor(
            tracker_executor,
            tracker.track_offline,
            video_path,
            bbox_parsed
        )
    finally:  
        def _cleanup_file(path_str: str):
            path = Path(path_str)
            if path.exists():
                path.unlink()

        # Delete temp file via I/O executor
        await loop.run_in_executor(io_executor, _cleanup_file, video_path)

    return {
        "status": "done",
        "metrics": metrics
    }