# Deepfake Detection API

## DeepfakeDetector

Unified deepfake detector interface.

### Constructor

```python
DeepfakeDetector(
    model_name: str = "resnet_inception",
    confidence_threshold: float = 0.5,
    device: str = None,
)
```

**Parameters:**
- `model_name`: Name of the model ("resnet_inception", "efficientnet")
- `confidence_threshold`: Minimum confidence threshold for detections
- `device`: Device to run inference on ("cpu" or "cuda")

### Methods

```python
detect(
    media_path: str,
    save_csv: bool = False,
    csv_path: str = "deepfake_detections.csv",
    save_annotated: bool = False,
    output_folder: str = "output",
    num_frames: int = 11,
    ) -> Union[DeepfakeDetection, List[DeepfakeDetection]]
```

Detects deepfake in the given media file (image or video).

**Parameters:**  
- `media_path`: Path to the input media file (image or video)  
- `save_csv`: Whether to save detection results to CSV file  
- `csv_path`: Path where to save the CSV file  
- `save_annotated`: Whether to save annotated media with results  
- `output_folder`: Folder path where to save annotated media  
- `num_frames`: Number of equally spaced frames to analyze for videos  

**Returns:**
- DeepfakeDetection and final_result

```python
get_model_info() -> dict
```

**Returns** information about the current model.

## Available Models

- `resnet_inception`
- `efficientnet` 