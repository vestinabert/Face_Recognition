from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
import io
from retinaface import RetinaFace
from deepface import DeepFace

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "server/uploads"
OUTPUT_FOLDER = "server/processed"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def detect_faces(image):
    """Detect faces in an image using RetinaFace."""
    return RetinaFace.detect_faces(image)


def identify_faces(input_image, detected_faces, reference_image):
    """Compare detected faces against the reference image."""
    matching_faces = {}
    non_matching_faces = {}

    for face_id, face_data in detected_faces.items():
        x1, y1, x2, y2 = face_data["facial_area"]
        face_crop = input_image[y1:y2, x1:x2]

        try:
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
        except Exception as e:
            print(f"Error comparing face {face_id}: {e}")
            non_matching_faces[face_id] = face_data

    return matching_faces, non_matching_faces


def blur_faces(image, faces_to_blur):
    """Blur all non-matching faces in the image."""
    for face_id, face_data in faces_to_blur.items():
        x1, y1, x2, y2 = face_data["facial_area"]
        face_roi = image[y1:y2, x1:x2]
        blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
        image[y1:y2, x1:x2] = blurred_face
    return image


@app.route("/process-image", methods=["POST"])
def process_image():
    """Process an input image against a reference image and save output in server/output/."""
    if "input" not in request.files or "reference" not in request.files:
        return jsonify({"error": "Both input and reference images are required"}), 400

    reference_file = request.files["reference"]
    reference_path = os.path.join(UPLOAD_FOLDER, "reference.jpg")
    reference_file.save(reference_path)
    reference_image = cv2.imread(reference_path)

    input_file = request.files["input"]
    input_path = os.path.join(UPLOAD_FOLDER, "input.jpg")
    input_file.save(input_path)
    input_image = cv2.imread(input_path)

    if reference_image is None or input_image is None:
        return jsonify({"error": "Invalid image format"}), 400

    detected_faces = detect_faces(input_path)

    if not detected_faces:
        return jsonify({"error": "No faces detected"}), 400

    matching_faces, non_matching_faces = identify_faces(
        input_image, detected_faces, reference_image
    )
    processed_image = blur_faces(input_image, non_matching_faces)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    output_path = os.path.join(OUTPUT_FOLDER, "output.jpg")
    success = cv2.imwrite(output_path, processed_image)

    if not success:
        return jsonify({"error": "Failed to save processed image"}), 500

    return send_file(output_path, mimetype="image/jpeg")


@app.route("/")
def home():
    return jsonify({"message": "Face Recognition API is running"})


if __name__ == "__main__":
    app.run(debug=True)
