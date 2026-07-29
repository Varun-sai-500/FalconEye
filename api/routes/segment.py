import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, UploadFile, Form, File, HTTPException, status
from pydantic import BaseModel
from typing import Optional
import cv2
import numpy as np
import json
from enum import Enum

from services.segmentation_service import sam_service, clipseg_service

router = APIRouter()

# Global worker thread to serialize compute and isolate the CUDA context
executor = ThreadPoolExecutor(max_workers=1)

class SegmentMethod(str, Enum):
    click = "click"
    reference = "reference"
    text = "text"

class SegmentResponse(BaseModel):
    bbox: tuple[int, int, int, int]

# ----------------------------------------------------------------------
# Pure CPU/GPU Worker (Executes entirely off the main event loop)
# ----------------------------------------------------------------------

def process_segmentation(
    method: SegmentMethod,
    frame_bytes: bytes,
    ref_bytes: Optional[bytes],
    points_str: Optional[str],
    text_str: Optional[str]
):
    # 1. Decode base image
    frame_bgr = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("Invalid input image encoding.")

    # 2. Decode reference image if present
    ref_bgr = None
    if ref_bytes and len(ref_bytes) > 0:
        ref_bgr = cv2.imdecode(np.frombuffer(ref_bytes, np.uint8), cv2.IMREAD_COLOR)
        if ref_bgr is None:
            raise ValueError("Invalid reference image encoding.")

    # 3. Structural Payload Validation
    point_list = None
    if method == SegmentMethod.click:
        if not points_str:
            raise ValueError("Points are required for click mode.")
        try:
            point_list = json.loads(points_str)
        except json.JSONDecodeError:
            raise ValueError("Invalid points JSON format.")

    elif method == SegmentMethod.reference:
        if ref_bytes is None or len(ref_bytes) == 0:
            raise ValueError("Reference image is required for reference mode.")

    elif method == SegmentMethod.text:
        if not text_str or not text_str.strip():
            raise ValueError("Non-empty text string is required for text mode.")

    # 4. Color space transformation
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 5. Model Inference Core Pass
    if method == SegmentMethod.click:
        bbox = sam_service.predict(rgb_frame, point_list=point_list)

    elif method == SegmentMethod.reference:
        ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
        bbox = clipseg_service.predict(rgb_frame, ref_image=ref_rgb)

    elif method == SegmentMethod.text:
        bbox = clipseg_service.predict(rgb_frame, text=text_str.strip())

    return bbox


# ----------------------------------------------------------------------
# Fluid Async Network Layer
# ----------------------------------------------------------------------

@router.post("/segment", response_model=SegmentResponse)
async def segment(
    file: UploadFile = File(...),
    method: SegmentMethod = Form(...),
    points: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    ref_file: Optional[UploadFile] = File(None),
):
    loop = asyncio.get_running_loop()

    # Safety wrapper check against empty structural form items
    try:
        file_bytes = await file.read()

        # Safely extract bytes from optional reference objects without event loop blocking
        ref_bytes = None
        if ref_file and ref_file.filename:
            ref_bytes = await ref_file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file payload components: {str(e)}"
        )

    try:
        bbox = await loop.run_in_executor(
            executor,
            process_segmentation,
            method,
            file_bytes,
            ref_bytes,
            points,
            text
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution engine fault: {str(e)}"
        )
    finally:
        # Clean up temporary spool descriptors immediately to clear RAM leaks
        await file.close()
        if ref_file:
            await ref_file.close()

    return SegmentResponse(
        bbox=bbox
    )