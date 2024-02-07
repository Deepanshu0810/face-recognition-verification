from fastapi import FastAPI
from api.routers import face_recognition, face_verification, live_face_recognition, video_face_recognition

api = FastAPI()

api.include_router(face_recognition.router)
api.include_router(face_verification.router)
api.include_router(live_face_recognition.router)
api.include_router(video_face_recognition.router)
