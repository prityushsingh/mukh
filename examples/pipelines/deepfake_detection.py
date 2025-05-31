import argparse

from mukh.pipelines.deepfake_detection import DeepfakeDetectionPipeline

parser = argparse.ArgumentParser(description="Deepfake Detection Pipeline")
parser.add_argument(
    "--media_path", type=str, required=True, help="Path to the media file to analyze"
)
args = parser.parse_args()

# Configure models
model_configs = [
    {
        "name": "resnet_inception",
        "confidence_threshold": 0.4,
    },
    {
        "name": "resnext",
        "model_variant": "resnext",
        "confidence_threshold": 0.5,
    },
    {
        "name": "efficientnet",
        "net_model": "EfficientNetB4",
        "confidence_threshold": 0.6,
    },
]

# Create pipeline with weighted averaging
model_weights = {"resnet_inception": 0.4, "resnext": 0.3, "efficientnet": 0.3}

pipeline = DeepfakeDetectionPipeline(
    model_configs=model_configs, model_weights=model_weights, confidence_threshold=0.5
)

# Detect deepfakes
result = pipeline.detect(
    media_path=args.media_path,
    save_csv=True,
    save_annotated=True,
    save_individual_results=True,
)
