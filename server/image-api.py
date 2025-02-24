from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
from retinaface import RetinaFace
from deepface import DeepFace

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

REFERENCE_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, "reference.jpg")  # Reference face


# Function to detect faces
def detect_faces(image_path):
    return RetinaFace.detect_faces(image_path)


# Function to identify faces
def identify_faces(input_image, detected_faces, reference_image):
    matching_faces = {}
    non_matching_faces = {}

    for face_id, face_data in detected_faces.items():
        x1, y1, x2, y2 = face_data["facial_area"]
        face_crop = input_image[y1:y2, x1:x2]

        result = DeepFace.verify(
            reference_image,
            face_crop,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=False,
        )

        if result["verified"]:
            matching_faces[face_id] = face_data
        else:
            non_matching_faces[face_id] = face_data

    return matching_faces, non_matching_faces


# Function to blur non-matching faces
def blur_faces(image, faces_to_blur):
    for face_id, face_data in faces_to_blur.items():
        x1, y1, x2, y2 = face_data["facial_area"]
        face_roi = image[y1:y2, x1:x2]
        blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
        image[y1:y2, x1:x2] = blurred_face
    return image


# API endpoint to process images
@app.route("/process-image", methods=["POST"])
def process_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    input_image = cv2.imread(filename)

    detected_faces = detect_faces(filename)
    if not detected_faces:
        return jsonify({"error": "No faces detected"}), 400

    matching_faces, non_matching_faces = identify_faces(
        input_image, detected_faces, REFERENCE_IMAGE_PATH
    )
    processed_image = blur_faces(input_image, non_matching_faces)

    output_path = os.path.join(PROCESSED_FOLDER, "output.jpg")
    cv2.imwrite(output_path, processed_image)

    return send_file(output_path, mimetype="image/jpeg")


@app.route("/")
def home():
    return jsonify({"message": "Face Recognition API is running"})


if __name__ == "__main__":
    app.run(debug=True)
