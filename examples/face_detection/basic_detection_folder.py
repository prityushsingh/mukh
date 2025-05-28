"""
Basic example showing how to use a single face detector.
"""

import argparse
import os
import shutil

import cv2

from mukh.face_detection import FaceDetector

# Set up argument parser
parser = argparse.ArgumentParser(description="Face Detection Example")
parser.add_argument(
    "--detection_model",
    type=str,
    choices=["blazeface", "mediapipe", "ultralight"],
    default="mediapipe",
    help="Choose the face detection model to use.",
)
parser.add_argument(
    "--clear_output",
    type=bool,
    default=False,
    help="Clear the output folder before saving new images.",
)
args = parser.parse_args()

# Clear output folder if specified
if args.clear_output and os.path.exists("output_images"):
    shutil.rmtree("output_images")
os.makedirs("output_images", exist_ok=True)

# Create detector
detector = FaceDetector.create(args.detection_model)

# Process all images in the demo_images folder
demo_images_folder = "demo_images"
for image_name in os.listdir(demo_images_folder):
    if image_name.endswith((".jpg", ".png")):
        # Get image path
        image_path = os.path.join(demo_images_folder, image_name)

        # Detect faces
        faces, annotated_image = detector.detect_with_landmarks(image_path)

        # Save output
        output_image_path = os.path.join(
            "output_images", f"{args.detection_model}_output_{image_name}"
        )
        cv2.imwrite(output_image_path, annotated_image)

        # Print results
        print(f"Found {len(faces)} faces in {image_name}")
        for face in faces:
            print(f"Face confidence: {face.bbox.confidence:.2f}")
