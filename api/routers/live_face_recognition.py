from fastapi import APIRouter
from training.facenet import FacenetModel
from fastapi.responses import StreamingResponse

router = APIRouter()
facenet_model = FacenetModel()

@router.get("/live_face_recognition/")
async def live_face_recognition():
    return StreamingResponse(facenet_model.recognize_live(), media_type="multipart/x-mixed-replace; boundary=frame")
