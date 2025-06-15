# Pipeline Deepfake Detection API

## PipelineDeepfakeDetection

Ensemble deepfake detection pipeline that combines multiple models with weighted averaging.

### Constructor

```python
PipelineDeepfakeDetection(
    model_configs: Dict[str, float],
    device: Optional[str] = None,
    confidence_threshold: float = 0.5,
)
```

**Parameters:**  
- `model_configs`: Dictionary mapping model names to their weights (e.g., `{"resnet_inception": 0.5, "efficientnet": 0.5}`)  
- `device`: Device to run inference on ("cpu" or "cuda"). Auto-detected if None  
- `confidence_threshold`: Threshold for ensemble prediction (default: 0.5)  

### Methods

```python
detect(
    media_path: str,
    output_folder: str,
    save_csv: bool = True,
    num_frames: int = 11,
) -> bool
```

Runs the complete ensemble deepfake detection pipeline.

**Parameters:**  
- `media_path`: Path to the media file (image or video) to analyze  
- `output_folder`: Folder path to save all detection results  
- `save_csv`: Whether to save detection results to CSV files  
- `num_frames`: Number of equally spaced frames for video analysis  

**Returns:**
- Final ensemble prediction (True for deepfake, False for real)

```python
get_model_info() -> Dict
```

**Returns:**
- Dictionary containing detector information (model_configs, device, confidence_threshold, total_models)

## Available Models

- `resnet_inception`
- `efficientnet`

## Example Usage

```python
from mukh.pipelines.deepfake_detection import PipelineDeepfakeDetection

# Define model configurations with weights
model_configs = {"resnet_inception": 0.5, "efficientnet": 0.5}

# Create ensemble detector
detector = PipelineDeepfakeDetection(model_configs)

# Run detection
result = detector.detect(
    media_path="path/to/media.mp4",
    output_folder="output/pipeline",
    save_csv=True,
    num_frames=11
)

print(f"Final Result: {'DEEPFAKE' if result else 'REAL'}")
``` 