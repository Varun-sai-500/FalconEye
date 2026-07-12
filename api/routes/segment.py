from fastapi import APIRouter, UploadFile, Form
from pydantic import BaseModel
from typing import Optional
import cv2
import numpy as np
import base64
import json
from fastapi import File
from fastapi import HTTPException
from enum import Enum

from services.segmentation_service import sam_service, clipseg_service
from core.utils.image_preprocessing import preprocess_frame
from core.utils.boundingbox import get_boundary

router = APIRouter()

class SegmentMethod(str, Enum):
    click = "click"
    reference = "reference"
    text = "text"

def validate_request(
    frame_bgr: np.ndarray,
    method: SegmentMethod,
    points: Optional[str],
    text: Optional[str],
    ref_file: Optional[UploadFile],
    ref_bgr: Optional[np.ndarray],
) -> list | None:
    if frame_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid input image."
        )

    if method == SegmentMethod.click:
        if not points:
            raise HTTPException(
                status_code=400,
                detail="Points are required for click mode."
            )

        try:
            point_list = json.loads(points)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid points JSON."
            )

        return point_list

    if method == SegmentMethod.reference:
        if ref_file is None:
            raise HTTPException(
                status_code=400,
                detail="Reference image is required for reference mode."
            )

        if ref_bgr is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid reference image."
            )

    if method == SegmentMethod.text:
        if not text or not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text is required for text mode."
            )

    return None


class SegmentResponse(BaseModel):
    mask_b64: str
    bbox: Optional[tuple[int, int, int, int]] = None

def encode_mask(mask: np.ndarray) -> str:
    mask = (mask.astype(np.uint8) * 255)
    _, buf = cv2.imencode(".png", mask)
    return base64.b64encode(buf).decode("utf-8")

def decode_image(file_bytes: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

@router.post("/segment", response_model=SegmentResponse)
async def segment(
    file: UploadFile,
    method: SegmentMethod = Form(...),          # "click" | "reference" | "text"
    points: Optional[str] = Form(None),  # JSON "[[x,y],...]" for click
    text: Optional[str] = Form(None),     # for text mode
    ref_file: Optional[UploadFile] = File(None),  # for reference mode
):
    frame_bgr = decode_image(await file.read())
    ref_bgr = None
    if ref_file is not None:
        ref_bgr = decode_image(await ref_file.read())

    point_list = validate_request(
        frame_bgr=frame_bgr,
        method=method,
        points=points,
        text=text,
        ref_file=ref_file,
        ref_bgr=ref_bgr,
    )

    rgb_frame, frame_resized = preprocess_frame(frame_bgr)

    if method == SegmentMethod.click:
        mask = sam_service.predict_points(rgb_frame, point_list)

    elif method == SegmentMethod.reference:
        ref_rgb, _ = preprocess_frame(ref_bgr)
        mask = clipseg_service.predict(rgb_frame, ref_image=ref_rgb)

    elif method == SegmentMethod.text:
        mask = clipseg_service.predict(rgb_frame, text=text.strip())
    bbox, _ = get_boundary(mask, frame_resized)

    return SegmentResponse(
        mask_b64=encode_mask(mask),
        bbox=bbox if bbox else None
    )