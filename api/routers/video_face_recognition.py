from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from training.facenet import FacenetModel
from api.dependencies import encrypt_file

router = APIRouter()
facenet_model = FacenetModel()

@router.post("/video_face_recognition/")
async def video_face_recognition(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        encrypted_contents = encrypt_file(contents)
        video_path = "temp.mp4"
        with open(video_path, "wb") as f:
            f.write(encrypted_contents)
        
        return StreamingResponse(facenet_model.video_recognition(video_path), media_type="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
