# Deepfake Detection Examples

## Basic Detection

```python
import torch
from mukh.deepfake_detection import DeepfakeDetector

detector = DeepfakeDetector(
    model_name="resnet_inception",  # Options: "resnet_inception", "efficientnet"
    confidence_threshold=0.5,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

detections, final_result = detector.detect(
    media_path="assets/images/img1.jpg",
    save_csv=True,
    csv_path="output/resnet_inception/deepfake_detections.csv",
    save_annotated=True,
    output_folder="output/resnet_inception",
    num_frames=11  # For video analysis
)

print(f"Deepfake: {final_result}")
```

## Available Models

- `resnet_inception`
- `efficientnet`

## Parameters

- `confidence_threshold`: Detection confidence threshold (0.0 to 1.0)
- `num_frames`: Number of frames to analyze for videos
- `media_path`: Path to image or video file 
- `num_frames`: Number of frames to analyze in the video