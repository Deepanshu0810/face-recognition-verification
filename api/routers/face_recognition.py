from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from training.facenet import FacenetModel
from api.dependencies import encrypt_file

router = APIRouter()
facenet_model = FacenetModel()

@router.post("/face_recognition/")
async def face_recognition(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        encrypted_contents = encrypt_file(contents)
        img_path = "api/temp/temp.jpg"
        with open(img_path, "wb") as f:
            f.write(encrypted_contents)
        
        img, _, _ = facenet_model.recognize(img_path)
        return StreamingResponse(content=img.tobytes(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
