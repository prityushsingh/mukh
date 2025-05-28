"""
Basic example showing how to use a single face detector to detect faces in a folder of images.

Usage:
python -m examples.face_detection.basic_detection_folder --detection_model <detection_model>
"""

import argparse
import os

from mukh.face_detection import FaceDetector

parser = argparse.ArgumentParser(description="Face Detection Example")
parser.add_argument(
    "--detection_model",
    type=str,
    choices=["blazeface", "mediapipe", "ultralight"],
    default="mediapipe",
    help="Choose the face detection model to use.",
)

args = parser.parse_args()

# Create detector
detector = FaceDetector.create(args.detection_model)

# Process all images in the demo_images folder
images_folder = "assets/images"
for image_name in os.listdir(images_folder):
    if image_name.endswith((".jpg", ".png")):
        # Get image path
        image_path = os.path.join(images_folder, image_name)

        # Detect faces
        detections = detector.detect(
            image_path=image_path,  # Path to the image to detect faces in
            save_csv=True,  # Save the detections to a CSV file
            csv_path=f"output/{args.detection_model}/detections.csv",  # Path to save the CSV file
            save_annotated=True,  # Save the annotated image
            output_folder=f"output/{args.detection_model}",  # Path to save the annotated image
        )
