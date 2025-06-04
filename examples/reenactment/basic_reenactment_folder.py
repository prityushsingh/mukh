"""
Basic example showing how to use a single face reenactor to reenact faces from a folder of images.

Usage:
python -m examples.reenactment.basic_reenactment_folder --reenactor_model <reenactor_model>
"""

import argparse
import os

from mukh.reenactment import FaceReenactor

parser = argparse.ArgumentParser(description="Face Reenactment Example")
parser.add_argument(
    "--reenactor_model",
    type=str,
    choices=["tps"],
    default="tps",
    help="Choose the reenactor model to use.",
)
parser.add_argument(
    "--source_folder_path",
    type=str,
    default="assets/images",
    help="Path to the source folder.",
)
parser.add_argument(
    "--driving_video_path",
    type=str,
    default="assets/videos/video_1sec.mp4",
    help="Path to the driving video.",
)
parser.add_argument(
    "--output_folder",
    type=str,
    default="output",
    help="Path to save the reenacted video.",
)

args = parser.parse_args()

# Create reenactor
reenactor = FaceReenactor.create(args.reenactor_model)

# Process all images in the demo_images folder
for image_name in os.listdir(args.source_folder_path):
    if image_name.endswith((".jpg", ".png")):
        # Get image path
        image_path = os.path.join(args.source_folder_path, image_name)
        image_basename = os.path.splitext(image_name)[0]

        # Reenact from video
        result_path = reenactor.reenact_from_video(
            source_path=image_path,  # Path to the image to reenact
            driving_video_path=args.driving_video_path,
            output_path=f"{args.output_folder}/{args.reenactor_model}/{image_basename}_reenacted",  # Save each reenacted video
            save_comparison=True,
            resize_to_image_resolution=False,
        )

        print(f"Reenacted video saved to: {result_path}")
