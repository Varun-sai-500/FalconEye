from fastapi import APIRouter
from pydantic import BaseModel
from services.following_service import follower_service

router = APIRouter()

class BBoxInput(BaseModel):
    bbox: list   # [x, y, w, h]

class CommandResponse(BaseModel):
    linear:  float
    angular: float
    state:   str
    error_x: float
    error_y: float
    bbox_h:  int

@router.post("/follow/command", response_model=CommandResponse)
def follow_command(body: BBoxInput):
    result = follower_service.compute(tuple(body.bbox))
    return CommandResponse(**result)

