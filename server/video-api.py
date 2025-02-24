from flask import Flask, request, jsonify, send_file
import os
import cv2
import face_recognition
import mediapipe as mp
import numpy as np

app = Flask(__name__)
SERVER_FOLDER = os.path.abspath("server")
UPLOAD_FOLDER = os.path.join(SERVER_FOLDER, "uploads")
OUTPUT_FOLDER = os.path.join(SERVER_FOLDER, "processed")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def load_reference_faces(reference_files):
    reference_encodings = []
    reference_names = []

    for file in reference_files:
        image = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            reference_encodings.append(encodings[0])
            reference_names.append(os.path.basename(file))

    return reference_encodings, reference_names


mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.6)


def process_video(video_path, output_path, reference_encodings, reference_names):
    print(f"Processing video: {video_path}")
    print(f"Saving output to: {output_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return False

    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
    )

    if not out.isOpened():
        print(f"Error: Unable to create output file {output_path}")
        return False

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
                        name = reference_names[best_match_idx]

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

    if not os.path.exists(output_path):
        print(f"Error: Processed video file {output_path} was not created.")
        return False
    return True


def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


@app.route("/process_video", methods=["POST"])
def upload_video():
    reference_files = request.files.getlist("references")
    video_file = request.files.get("video")

    if not reference_files:
        return jsonify({"error": "No reference images uploaded"}), 400

    if not video_file or not allowed_file(video_file.filename, {"mp4", "avi", "mov"}):
        return jsonify({"error": "Invalid video file format"}), 400

    reference_paths = []
    for ref_file in reference_files:
        ref_path = os.path.join(UPLOAD_FOLDER, ref_file.filename)
        ref_file.save(ref_path)
        reference_paths.append(ref_path)

    reference_encodings, reference_names = load_reference_faces(reference_paths)

    if not reference_encodings:
        return jsonify({"error": "No faces found in reference images"}), 400

    input_video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    output_video_path = os.path.join(OUTPUT_FOLDER, f"processed_{video_file.filename}")
    video_file.save(input_video_path)

    success = process_video(
        input_video_path, output_video_path, reference_encodings, reference_names
    )
    if not success or not os.path.exists(output_video_path):
        return jsonify({"error": "Processing failed. Output file not found."}), 500

    return send_file(output_video_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
