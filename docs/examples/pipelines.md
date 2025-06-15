# Pipeline Examples

## Deepfake Detection Pipeline

```python
from mukh.pipelines.deepfake_detection import PipelineDeepfakeDetection

# Define model configurations with weights
model_configs = {"resnet_inception": 0.5, "efficientnet": 0.5}

# Create ensemble detector
detector = PipelineDeepfakeDetection(model_configs)

# Run detection
result = detector.detect(
    media_path="assets/images/img1.jpg",
    output_folder="output/deepfake_detection_pipeline",
    save_csv=True,
    num_frames=11
)

print(f"Final Result: {'DEEPFAKE' if result else 'REAL'}")
```

## Model Information

```python
# Print detector information
info = detector.get_model_info()
for key, value in info.items():
    print(f"  {key}: {value}")
```

## Available Models

- `resnet_inception`
- `efficientnet`

## Parameters

- `model_configs`: Dictionary of model names and their weights
- `media_path`: Path to image or video file
- `output_folder`: Output directory for results
- `save_csv`: Save detection results to CSV
- `num_frames`: Number of frames to analyze for videos