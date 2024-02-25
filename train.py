from training.facenet import FacenetModel
import os
import cv2 as cv
import pickle
from pathlib import Path

model = FacenetModel()

face_data = Path('new_data')

print("###### Training new faces ######")
model.train_new(face_data)
print("###### Training complete ######")