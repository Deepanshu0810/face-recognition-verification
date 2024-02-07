from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from training.facenet import FacenetModel
from api.dependencies import encrypt_file
from typing import List

router = APIRouter()
facenet_model = FacenetModel()

@router.post("/face_verification/")
async def face_verification(files: List[UploadFile]):
    try:
        if len(files) != 2:
            raise HTTPException(status_code=400, detail="Exactly two images are required for verification")
        
        img1_contents = await files[0].read()
        img2_contents = await files[1].read()
        encrypted_img1_contents = encrypt_file(img1_contents)
        encrypted_img2_contents = encrypt_file(img2_contents)
        img1_path = "temp1.jpg"
        img2_path = "temp2.jpg"
        
        with open(img1_path, "wb") as f1, open(img2_path, "wb") as f2:
            f1.write(encrypted_img1_contents)
            f2.write(encrypted_img2_contents)
        
        img1, img2 = facenet_model.verify(img1_path, img2_path)
        return StreamingResponse(content=img1.tobytes(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
