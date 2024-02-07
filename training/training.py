from model import InceptionResNetV2
import os 
import cv2
import mtcnn
import pickle 
import numpy as np 
from sklearn.preprocessing import Normalizer
from keras.models import load_model
from glob import glob

######pathsandvairables#########
face_data = '../data/105_classes_pins_dataset'
required_shape = (160,160)
face_encoder = InceptionResNetV2()
path = "../model/keras-facenet/weights/facenet_keras_weights.h5"
face_encoder.load_weights(path)
face_detector = mtcnn.MTCNN()
encodes = []
encoding_dict = dict()
l2_normalizer = Normalizer('l2')
###############################


def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def train():
    for person in os.listdir(face_data):
        name = person.split("_")[-1].title()
        person_dir = os.path.join(face_data,person)
        filepaths = np.random.choice(glob(person_dir + '/*'), size=10)
        encodes = []

        for image_name in filepaths:
            image_path = os.path.join(person_dir,image_name)

            img_BGR = cv2.imread(image_path)
            img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

            x = face_detector.detect_faces(img_RGB)
            if len(x) > 0:
                x1, y1, width, height = x[0]['box']
                x1, y1 = abs(x1) , abs(y1)
                x2, y2 = x1+width , y1+height
                face = img_RGB[y1:y2 , x1:x2]
            else:
                face = img_RGB

            face = normalize(face)
            face = cv2.resize(face, required_shape)
            face_d = np.expand_dims(face, axis=0)
            encode = face_encoder.predict(face_d)[0]
            encodes.append(encode)

    if encodes:
        encode = np.sum(encodes, axis=0 )
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        encoding_dict[name] = encode






