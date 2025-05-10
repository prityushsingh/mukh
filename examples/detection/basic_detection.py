"""
Basic example showing how to use a single face detector.
"""

import os

import cv2

from mukh.detection import FaceDetector

detection_model = "mediapipe"

# Create detector
detector = FaceDetector.create(detection_model)

# To view available models
# print(detector.list_available_models())

# Process all images in the demo_images folder
demo_images_folder = "demo_images"
for image_name in os.listdir(demo_images_folder):
    if image_name.endswith((".jpg", ".png")):
        # Load image
        image_path = os.path.join(demo_images_folder, image_name)
        image = cv2.imread(image_path)

        # Detect faces
        faces, annotated_image = detector.detect_with_landmarks(image)

        # Save output
        output_image_path = os.path.join(
            "output_images", f"{detection_model}_output_{image_name}"
        )
        cv2.imwrite(output_image_path, annotated_image)

        # Print results
        print(f"Found {len(faces)} faces in {image_name}")
        for face in faces:
            print(f"Face confidence: {face.bbox.confidence:.2f}")
