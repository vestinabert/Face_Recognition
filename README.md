# Face Recognition API

## Overview
This project provides a face recognition system that consists of three major components:
1. **Image Processing API (Flask-based)** – Detects faces, identifies a specific person, and blurs non-matching faces in images.
2. **Video Processing API (Flask-based)** – Detects and recognizes faces in videos, labeling matched individuals.
3. **Hugging Face API Deployment** – A Gradio-based implementation of the image processing API, deployed on Hugging Face Spaces.
4. **Web Interface** – A basic HTML and JavaScript-based frontend for interacting with the API.

## Technologies Used
- **Flask** – For creating the backend API services.
- **DeepFace** – Used for face verification and identification.
- **RetinaFace** – Used for high-quality face detection in images.
- **Face Recognition Library** – Used for encoding and recognizing faces in videos.
- **OpenCV** – For image and video processing.
- **MediaPipe** – Used for additional face detection in video processing.
- **Gradio** – For deploying the API on Hugging Face.
---

# Part 1: Image Processing API
## Features
- Detects faces in an uploaded image.
- Identifies faces by comparing them with a reference face.
- Blurs all non-matching faces.

## Implementation
The API detects faces using RetinaFace, identifies them using DeepFace's ArcFace model, and blurs all non-matching faces using OpenCV. It receives an input image along with a reference image, processes it, and returns the modified image with non-matching faces blurred.

### Running the API
#### Install Dependencies
```bash
pip install flask flask-cors opencv-python numpy retinaface deepface
```
#### Start the API
```bash
python server/image-api.py
```
The API will start on `http://127.0.0.1:5000`. To process an image, send a POST request to `/process-image` with both an input image and a reference image.

#### cURL Command to Test
```bash
curl -X POST http://127.0.0.1:5000/process-image \
     -F "input=@path/to/your/input.jpg" \
     -F "reference=@path/to/your/reference.jpg" \
     --output processed_image.jpg
```
This command sends an input image and a reference image to the API. The system will detect faces, compare them with the reference image, blur non-matching faces, and return the processed image as `processed_image.jpg`. Ensure that the server is running before executing this command.

---

# Part 2: Video Processing API
## Features
- Detects faces in a video.
- Allows dynamic uploading of reference face images.
- Identifies people based on filenames of uploaded reference images.
- Labels recognized individuals and saves the processed video.

## Implementation
The video processing API dynamically loads reference images uploaded by the user. It extracts names from filenames and encodes faces using Face Recognition Library. During video processing, faces are detected using MediaPipe, matched against the uploaded reference images, and labeled accordingly before saving the processed video.

### Running the API
#### Install Dependencies
Ensure you have the required dependencies installed:
```bash
pip install flask opencv-python numpy face-recognition mediapipe
```
#### Start the API
Run the following command to start the video processing API:
```bash
python server/video-api.py
```
First, upload reference face images via the `/upload_references` endpoint. Then, upload a video via a POST request to `/process_video`. The system will detect and label faces based on the uploaded reference images, and return the processed video.

#### cURL Command to Test
```bash
curl -X POST http://127.0.0.1:5000/process_video \
     -F "references=@path/to/face1.jpg" \
     -F "references=@path/to/face2.jpg" \
     -F "video=@path/to/video.mp4" \
     --output processed_video.mp4
```
This will send reference images and a video file to the server, which will process the video and return the labeled version as `processed_video.mp4`. Ensure that the server is running before executing this command.


---

# Part 3: Hugging Face API Deployment
## Features
- Gradio-based interface for face recognition.
- Detects, identifies, and blurs non-matching faces.

## Implementation
The Gradio interface allows users to upload a reference image and an input image. The API processes the images and displays the output with non-matching faces blurred. The solution is hosted on Hugging Face Spaces.

### Running the API on Hugging Face
The API is hosted on Hugging Face Spaces and can be accessed online.
- Access the Hugging Face API here: [Hugging Face Face Recognition API](https://huggingface.co/spaces/Vestina/face-recognition-api)

---

# Part 4: Web Interface
- Basic HTML and JavaScript frontend that interacts with the Flask API.
- Future plans include upgrading to a React-based frontend.
- Access the web interface here: [Face Recognition Web Page](https://vestinabert.github.io/Face_Recognition/)

---

## Hosting with Ngrok
#### Install Ngrok
```bash
pip install pyngrok
```
#### Expose Flask API
```bash
ngrok http 5000
```
Ngrok creates a public URL that can be used to access the local Flask API remotely.


