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

if not reference_encodings:
    raise ValueError("No reference faces were encoded! Check image paths.")

# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.6)


# Open video file
VIDEO_PATH = "input_video.mp4"
OUTPUT_PATH = "output_video.mp4"
PROCESS_EVERY_N_FRAMES = 35  # Process every N frames

cap = cv2.VideoCapture(VIDEO_PATH)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# Define output video writer
out = cv2.VideoWriter(
    OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
)

frame_count = 0
start_time = time.time()
previous_faces = []  # Store last detected faces


def process_frame(rgb_frame):
    return face_detector.process(rgb_frame)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        thread = threading.Thread(target=process_frame, args=(rgb_frame,))
        thread.start()
        results = process_frame(rgb_frame)

        previous_faces.clear()
        if results and results.detections:
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(
                face_locations, face_encodings
            ):
                name = "Unknown"
                distances = face_recognition.face_distance(
                    reference_encodings, face_encoding
                )
                best_match_idx = np.argmin(distances) if distances.size > 0 else None

                if (
                    best_match_idx is not None
                    and face_recognition.compare_faces(
                        [reference_encodings[best_match_idx]], face_encoding
                    )[0]
                ):
                    name = REFERENCE_NAMES[best_match_idx]

                previous_faces.append((left, top, right, bottom, name))

    # Draw previous face locations
    for left, top, right, bottom, name in previous_faces:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    out.write(frame)
    frame_count += 1

    elapsed_time = time.time() - start_time
    progress = (frame_count / total_frames) * 100
    estimated_time_left = (
        (elapsed_time / frame_count) * (total_frames - frame_count)
        if frame_count > 0
        else 0
    )
    print(
        f"Processing: {progress:.2f}% | Time Left: {estimated_time_left:.2f}s", end="\r"
    )

cap.release()
out.release()
cv2.destroyAllWindows()
