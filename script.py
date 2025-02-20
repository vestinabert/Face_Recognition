import cv2
import numpy as np
import matplotlib.pyplot as plt
from retinaface import RetinaFace
from deepface import DeepFace


input_image_path = "input.jpg"
reference_image_path = "reference.jpg"

detected_faces = RetinaFace.detect_faces(input_image_path)
print(detected_faces)