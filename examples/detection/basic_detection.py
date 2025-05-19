import os

import cv2

from mukh.detection import FaceDetector

# Initialize detector
detection_model = (
    "blazeface"  # Available models: "blazeface", "mediapipe", "ultralight"
)
detector = FaceDetector.create(detection_model)

# Detect faces
image_path = "demo_images/1.jpg"
faces, annotated_image = detector.detect_with_landmarks(image_path)

# Save output
os.makedirs("output_images", exist_ok=True)
output_path = f"output_images/1.jpg"
cv2.imwrite(output_path, annotated_image)
