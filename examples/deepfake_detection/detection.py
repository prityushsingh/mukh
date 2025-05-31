"""
Basic example showing how to use deepfake detector to detect deepfakes in media files.

Usage:
python -m examples.deepfake_detection.detection
"""

import argparse

from mukh.deepfake_detection import DeepfakeDetector

parser = argparse.ArgumentParser(description="Deepfake Detection Example")
parser.add_argument(
    "--detection_model",
    type=str,
    choices=["resnet_inception", "resnext", "efficientnet"],
    default="resnet_inception",
    help="Choose the deepfake detection model to use.",
)
parser.add_argument(
    "--media_path",
    type=str,
    default="assets/images/img1.jpg",
    help="Path to the media file (image or video) to analyze for deepfakes.",
)
parser.add_argument(
    "--confidence_threshold",
    type=float,
    default=0.5,
    help="Confidence threshold for deepfake detection (0.0 to 1.0).",
)
parser.add_argument(
    "--num_frames",
    type=int,
    default=11,
    help="Number of equally spaced frames to extract from video for analysis.",
)

args = parser.parse_args()

# Create detector
detector = DeepfakeDetector(
    model_name=args.detection_model,
    confidence_threshold=args.confidence_threshold,
)

# Detect deepfakes in the media file
detections = detector.detect(
    media_path=args.media_path,  # Path to the media file to analyze
    save_csv=True,  # Save the detections to a CSV file
    csv_path=f"output/{args.detection_model}/deepfake_detections.csv",  # Path to save the CSV file
    save_annotated=True,  # Save the annotated media
    output_folder=f"output/{args.detection_model}",  # Path to save the annotated media
    num_frames=args.num_frames,  # Number of equally spaced frames for video analysis
)

# Print results
if isinstance(detections, list):
    print(f"Analyzed {len(detections)} frames from video")
    for i, detection in enumerate(detections):
        print(
            f"Frame {i}: {'Deepfake' if detection.is_deepfake else 'Real'} "
            f"(confidence: {detection.confidence:.3f})"
        )
else:
    print(
        f"Result: {'Deepfake' if detections.is_deepfake else 'Real'} "
        f"(confidence: {detections.confidence:.3f})"
    )
