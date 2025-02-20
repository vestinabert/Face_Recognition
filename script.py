import cv2
import numpy as np
import matplotlib.pyplot as plt
from retinaface import RetinaFace
from deepface import DeepFace


input_image_path = "input.jpg"
reference_image_path = "reference.jpg"

detected_faces = RetinaFace.detect_faces(input_image_path)
print(detected_faces)

matching_faces = {}
non_matching_faces = {}

for face_id, face_data in detected_faces.items():
    x1, y1, x2, y2 = face_data["facial_area"]

    img = cv2.imread(input_image_path)
    face_crop = img[y1:y2, x1:x2]
    detected_face_path = f"detected_{face_id}.jpg"
    cv2.imwrite(detected_face_path, face_crop)

    result = DeepFace.verify(
        reference_image_path,
        detected_face_path,
        model_name="ArcFace",
        detector_backend="retinaface",
    )

    if result["verified"]:
        matching_faces[face_id] = face_data
    else:
        non_matching_faces[face_id] = face_data
