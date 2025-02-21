import cv2
import face_recognition
import mediapipe as mp
import time
import threading
import numpy as np


REFERENCE_IMAGES = ["face1.jpg", "face2.jpg"]
REFERENCE_NAMES = ["Person 1", "Person 2"]

# Load reference images and extract their encodings
def load_reference_faces():
    reference_encodings = []
    for image_path in REFERENCE_IMAGES:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            reference_encodings.append(encodings[0])
    return reference_encodings

reference_encodings = load_reference_faces()
print(reference_encodings)