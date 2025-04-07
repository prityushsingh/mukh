"""
Basic example showing how to use a single face detector.
"""
import cv2
from mukh.detection import FaceDetector

detection_model = "mediapipe"

# Create detector
detector = FaceDetector.create(detection_model)

# To view available models
# print(detector.list_available_models())

# Load image
image = cv2.imread("demo_images/1.jpg")

# Detect faces
faces, annotated_image = detector.detect_with_landmarks(image)

# Save output
cv2.imwrite(f"output_images/{detection_model}_output.png", annotated_image)

# Print results
print(f"Found {len(faces)} faces")
for face in faces:
    print(f"Face confidence: {face.bbox.confidence:.2f}")
