from flask import Flask, request, jsonify, send_file
import os
import cv2
import face_recognition
import mediapipe as mp
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

REFERENCE_IMAGES = ["uploads/face1.jpg", "uploads/face2.jpg"]
REFERENCE_NAMES = ["Person 1", "Person 2"]


def load_reference_faces():
    reference_encodings = []
    for image_path in REFERENCE_IMAGES:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            reference_encodings.append(encodings[0])
    return reference_encodings


reference_encodings = load_reference_faces()
if not reference_encodings:
    raise ValueError("No reference faces were encoded! Check image paths.")

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.6)


def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
    )

    frame_count = 0
    previous_faces = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 35 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_frame)
            previous_faces.clear()

            if results and results.detections:
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                face_encodings = face_recognition.face_encodings(
                    rgb_frame, face_locations
                )

                for (top, right, bottom, left), face_encoding in zip(
                    face_locations, face_encodings
                ):
                    name = "Unknown"
                    distances = face_recognition.face_distance(
                        reference_encodings, face_encoding
                    )
                    best_match_idx = (
                        np.argmin(distances) if distances.size > 0 else None
                    )

                    if (
                        best_match_idx is not None
                        and face_recognition.compare_faces(
                            [reference_encodings[best_match_idx]], face_encoding
                        )[0]
                    ):
                        name = REFERENCE_NAMES[best_match_idx]

                    previous_faces.append((left, top, right, bottom, name))

        for left, top, right, bottom, name in previous_faces:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame,
                name,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {
        "mp4",
        "avi",
        "mov",
    }


@app.route("/process_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["video"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format"}), 400

    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(OUTPUT_FOLDER, "processed_" + file.filename)
    file.save(input_path)

    process_video(input_path, output_path)
    return send_file(output_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
