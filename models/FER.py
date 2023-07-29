import cv2
import os
# this code was tested in Python 3.10
from deepface import DeepFace

def load_face_detection_model():
    face_cascade = cv2.CascadeClassifier(os.environ['CHATBOT_ROOT'] + "/resources/haarcascade_frontalface_default.xml")
    return face_cascade

def detect_FER(face_cascade,frame):

    result = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)


    emotion = result[0]["dominant_emotion"]
    txt = str(emotion)
    return txt
