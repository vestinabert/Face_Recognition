import cv2
import numpy as np
import matplotlib.pyplot as plt
from retinaface import RetinaFace
from deepface import DeepFace

# File paths
INPUT_IMAGE_PATH = "input.jpg"
REFERENCE_IMAGE_PATH = "reference.jpg"


# Detect Faces
def detect_faces(image_path):
    return RetinaFace.detect_faces(image_path)


# Identify the Target Face
def identify_faces(input_image, detected_faces, reference_image):
    matching_faces = {}
    non_matching_faces = {}

    for face_id, face_data in detected_faces.items():
        x1, y1, x2, y2 = face_data["facial_area"]
        face_crop = input_image[y1:y2, x1:x2]

        # Verify face without saving the cropped face
        result = DeepFace.verify(
            reference_image,
            face_crop,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=False,  # Avoids errors if a face is not detected
        )

        if result["verified"]:
            matching_faces[face_id] = face_data
        else:
            non_matching_faces[face_id] = face_data

    return matching_faces, non_matching_faces


# Blur Non-Matching Faces
def blur_faces(image, faces_to_blur):
    for face_id, face_data in faces_to_blur.items():
        x1, y1, x2, y2 = face_data["facial_area"]
        face_roi = image[y1:y2, x1:x2]
        blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
        image[y1:y2, x1:x2] = blurred_face
    return image


def main():
    input_image = cv2.imread(INPUT_IMAGE_PATH)

    # Detect faces
    detected_faces = detect_faces(INPUT_IMAGE_PATH)

    if not detected_faces:
        print("No faces detected.")
        return

    print(f"Detected Faces: {detected_faces}")

    # Identify target face
    matching_faces, non_matching_faces = identify_faces(
        input_image, detected_faces, REFERENCE_IMAGE_PATH
    )

    # Blur non-matching faces
    processed_image = blur_faces(input_image, non_matching_faces)

    # Save & Display result
    output_path = "output.jpg"
    cv2.imwrite(output_path, processed_image)

    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
    print(f"Processed image saved as {output_path}")


if __name__ == "__main__":
    main()
