from .model import InceptionResNetV2
from scipy.spatial.distance import cosine
from sklearn.preprocessing import Normalizer
from app.constants import *
import mtcnn
import os
from glob import glob
import pickle
import cv2
import numpy as np

cwd = os.getcwd()

class FacenetModel:
    def __init__(
            self,
            data_dir = DATA_DIR,
            encodings_path = ENCODINGS_PATH,
            weights_path = WEIGHTS_PATH,
    ):
        self.required_shape = (160,160)
        self.face_encoder = InceptionResNetV2()
        self.path_m = weights_path
        self.face_encoder.load_weights(self.path_m)
        self.encodings_path = encodings_path
        self.face_data = data_dir
        self.face_detector = mtcnn.MTCNN()
        self.confidence_t=0.99
        self.recognition_t=0.5
        self.required_size = (160,160)
        self.encoding_dict = self.load_pickle()
        self.l2_normalizer = Normalizer('l2')

    def normalize(self,img):
        mean, std = img.mean(), img.std()
        return (img - mean) / std
    
    def get_face(self,img, box):
        x1, y1, width, height = box
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = img[y1:y2, x1:x2]
        return face, (x1, y1), (x2, y2)

    def get_encode(self, face, size):
        face = self.normalize(face)
        face = cv2.resize(face, size)
        encode = self.face_encoder.predict(np.expand_dims(face, axis=0))[0]
        return encode

    def load_pickle(self):
        with open(self.encodings_path, 'rb') as f:
            encoding_dict = pickle.load(f)
        return encoding_dict
    
    def train(self):
        for person in os.listdir(self.face_data):
            name = person.split("_")[-1].title()
            person_dir = os.path.join(self.face_data,person)
            filepaths = np.random.choice(glob(person_dir + '/*'), size=10)
            encodes = []

            for image_name in filepaths:
                image_path = os.path.join(person_dir,image_name)

                img_BGR = cv2.imread(image_path)
                img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

                x = self.face_detector.detect_faces(img_RGB)
                if len(x) > 0:
                    x1, y1, width, height = x[0]['box']
                    x1, y1 = abs(x1) , abs(y1)
                    x2, y2 = x1+width , y1+height
                    face = img_RGB[y1:y2 , x1:x2]
                else:
                    face = img_RGB

                face = self.normalize(face)
                face = cv2.resize(face, self.required_shape)
                face_d = np.expand_dims(face, axis=0)
                encode = self.face_encoder.predict(face_d)[0]
                encodes.append(encode)

        if encodes:
            encode = np.sum(encodes, axis=0 )
            encode = self.l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
            self.encoding_dict[name] = encode
    
    def recognize(self,img):
        img = cv2.imread(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detector.detect_faces(img_rgb)
        all_name = []
        all_distance = []
        for res in results:
            if res['confidence'] < self.confidence_t:
                continue
            face, pt_1, pt_2 = self.get_face(img_rgb, res['box'])
            encode = self.get_encode(face, self.required_size)
            encode = self.l2_normalizer.transform(encode.reshape(1, -1))[0]
            name = 'unknown'

            distance = float("inf")
            for db_name, db_encode in self.encoding_dict.items():
                dist = cosine(db_encode, encode)
                if dist < self.recognition_t and dist < distance:
                    name = db_name
                    distance = dist

            if name == 'unknown':
                cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
                cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            else:
                cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
                cv2.putText(img, name , (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 200), 2)

            all_name.append(name)
            all_distance.append(distance)
        
        return img, all_name, all_distance
    
    def verify(self,img1,img2):
        img1 = cv2.imread(img1)
        img2 = cv2.imread(img2)
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        results1 = self.face_detector.detect_faces(img1_rgb)
        results2 = self.face_detector.detect_faces(img2_rgb)
        
        for res1 in results1:
            for res2 in results2:
                if res1['confidence'] < self.confidence_t or res2['confidence'] < self.confidence_t:
                    continue
                face1, pt_11, pt_12 = self.get_face(img1_rgb, res1['box'])
                face2, pt_21, pt_22 = self.get_face(img2_rgb, res2['box'])
                encode1 = self.get_encode(face1, self.required_size)
                encode2 = self.get_encode(face2, self.required_size)
                encode1 = self.l2_normalizer.transform(encode1.reshape(1, -1))[0]
                encode2 = self.l2_normalizer.transform(encode2.reshape(1, -1))[0]
                distance = cosine(encode1, encode2)
                name = 'unknown'
                if distance < self.recognition_t:
                    name = 'verified'
                    cv2.rectangle(img1, pt_11, pt_12, (0, 255, 0), 2)
                    cv2.putText(img1, name , (pt_11[0], pt_11[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 200), 2)
                    cv2.rectangle(img2, pt_21, pt_22, (0, 255, 0), 2)
                    cv2.putText(img2, name , (pt_21[0], pt_21[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 200), 2)
                # else:
                #     name = 'unverified'
                #     cv2.rectangle(img1, pt_11, pt_12, (0, 0, 255), 2)
                #     cv2.putText(img1, name , (pt_11[0], pt_11[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 200), 2)
                #     cv2.rectangle(img2, pt_21, pt_22, (0, 0, 255), 2)
                #     cv2.putText(img2, name , (pt_21[0], pt_21[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 200), 2)

        return img1, img2

    def recognize_live(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detector.detect_faces(frame_rgb)
            for res in results:
                if res['confidence'] < self.confidence_t:
                    continue
                face, pt_1, pt_2 = self.get_face(frame_rgb, res['box'])
                encode = self.get_encode(face, self.required_size)
                encode = self.l2_normalizer.transform(encode.reshape(1, -1))[0]
                name = 'unknown'

                distance = float("inf")
                for db_name, db_encode in self.encoding_dict.items():
                    dist = cosine(db_encode, encode)
                    if dist < self.recognition_t and dist < distance:
                        name = db_name
                        distance = dist

                if name == 'unknown':
                    cv2.rectangle(frame, pt_1, pt_2, (0, 0, 255), 2)
                    cv2.putText(frame, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                else:
                    cv2.rectangle(frame, pt_1, pt_2, (0, 255, 0), 2)
                    cv2.putText(frame, name , (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 200), 2)

            # show camera feed in flask
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            

            # cv2.imshow('img', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        # cap.release()
        # cv2.destroyAllWindows()
            
    def video_recognition(self,video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detector.detect_faces(frame_rgb)
            for res in results:
                if res['confidence'] < self.confidence_t:
                    continue
                face, pt_1, pt_2 = self.get_face(frame_rgb, res['box'])
                encode = self.get_encode(face, self.required_size)
                encode = self.l2_normalizer.transform(encode.reshape(1, -1))[0]
                name = 'unknown'

                distance = float("inf")
                for db_name, db_encode in self.encoding_dict.items():
                    dist = cosine(db_encode, encode)
                    if dist < self.recognition_t and dist < distance:
                        name = db_name
                        distance = dist

                if name == 'unknown':
                    cv2.rectangle(frame, pt_1, pt_2, (0, 0, 255), 2)
                    cv2.putText(frame, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                else:
                    cv2.rectangle(frame, pt_1, pt_2, (0, 255, 0), 2)
                    cv2.putText(frame, name , (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 200), 2)

            # show processed video in flask
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            
        cap.release()