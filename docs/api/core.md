# Core Concepts

This section covers the fundamental concepts, data structures, and patterns used throughout Mukh.

## Detection Results

All detection modules return structured data objects that provide consistent access to results.

### Face Detection Results

Face detection methods return lists of detection objects with bounding box information:

```python
from mukh.face_detection import FaceDetector

detector = FaceDetector.create("mediapipe")
detections = detector.detect("image.jpg")

# Each detection contains bounding box and confidence
for detection in detections:
    print(f"Confidence: {detection.bbox.confidence}")
    print(f"Position: ({detection.bbox.x1}, {detection.bbox.y1}) to ({detection.bbox.x2}, {detection.bbox.y2})")
    print(f"Size: {detection.bbox.width} x {detection.bbox.height}")
    
    # Some models provide facial landmarks
    if hasattr(detection, 'landmarks') and detection.landmarks is not None:
        print(f"Landmarks: {len(detection.landmarks)} points")
```

### Deepfake Detection Results

Deepfake detection returns classification results with confidence scores:

```python
from mukh.deepfake_detection import DeepfakeDetector

detector = DeepfakeDetector("efficientnet")
result = detector.detect("image.jpg")

# Single image result
print(f"Is deepfake: {result.is_deepfake}")
print(f"Confidence: {result.confidence}")
print(f"Model used: {result.model_name}")

# Video results (list of frame results)
video_results = detector.detect("video.mp4", num_frames=10)
for frame_result in video_results:
    print(f"Frame {frame_result.frame_number}: {frame_result.is_deepfake} (conf: {frame_result.confidence})")
```

## Common Patterns

### Model Selection

All modules use a factory pattern for model creation:

```python
# Face Detection - choose from blazeface, mediapipe, ultralight
face_detector = FaceDetector.create("mediapipe")

# Deepfake Detection - choose from resnet_inception, resnext, efficientnet  
deepfake_detector = DeepfakeDetector("efficientnet")

# Face Reenactment - currently supports tps
reenactor = FaceReenactor.create("tps")
```

### File I/O Patterns

All modules support flexible input/output options:

```python
# Basic detection with file paths
detections = detector.detect(
    image_path="input.jpg",
    save_csv=True,
    csv_path="results.csv",
    save_annotated=True,
    output_folder="output"
)

# Video processing with frame sampling
results = detector.detect(
    media_path="video.mp4",
    num_frames=15,  # Sample 15 frames
    output_folder="video_analysis"
)
```

### Batch Processing

Process multiple files efficiently:

```python
import os
from mukh.face_detection import FaceDetector

detector = FaceDetector.create("mediapipe")

# Process all images in a directory
image_dir = "input_images"
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_dir, filename)
        
        detections = detector.detect(
            image_path=image_path,
            save_csv=True,
            output_folder=f"output/{filename.split('.')[0]}"
        )
        print(f"{filename}: {len(detections)} detections")
```

## Configuration Patterns

### Confidence Thresholds

Control detection sensitivity across modules:

```python
# Face detection with custom confidence
detector = FaceDetector.create("blazeface")
# Note: confidence filtering happens in post-processing

# Deepfake detection with threshold
deepfake_detector = DeepfakeDetector("resnext", confidence_threshold=0.7)

# Check if result meets threshold
result = deepfake_detector.detect("image.jpg")
if result.confidence >= 0.7:
    print(f"High confidence detection: {result.is_deepfake}")
```

### Output Customization

Control what gets saved and where:

```python
# Minimal output - just get results
detections = detector.detect("image.jpg")

# Save CSV results only
detections = detector.detect(
    "image.jpg",
    save_csv=True,
    csv_path="detections.csv"
)

# Save annotated images with custom folder
detections = detector.detect(
    "image.jpg", 
    save_annotated=True,
    output_folder="annotated_results"
)

# Full output - save everything
detections = detector.detect(
    "image.jpg",
    save_csv=True,
    csv_path="detections.csv", 
    save_annotated=True,
    output_folder="full_results"
)
```

## Error Handling

### Input Validation

```python
import os
from mukh.face_detection import FaceDetector

def safe_detection(image_path):
    """Safely perform face detection with validation."""
    # Validate file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Validate file format
    valid_formats = ('.jpg', '.jpeg', '.png', '.bmp')
    if not image_path.lower().endswith(valid_formats):
        raise ValueError(f"Unsupported format. Use: {valid_formats}")
    
    try:
        detector = FaceDetector.create("mediapipe")
        return detector.detect(image_path)
    except Exception as e:
        print(f"Detection failed: {e}")
        return []
```

### Graceful Degradation

```python
def robust_analysis(media_path):
    """Analyze media with fallback options."""
    # Try multiple models in order of preference
    models = ["efficientnet", "resnet_inception", "resnext"]
    
    for model_name in models:
        try:
            detector = DeepfakeDetector(model_name)
            return detector.detect(media_path)
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            continue
    
    raise RuntimeError("All models failed")
```

## Performance Considerations

### Model Selection Guide

| Use Case | Recommended Model | Trade-offs |
|----------|------------------|------------|
| **Face Detection** | | |
| Real-time/Mobile | `blazeface` | Fast, lower accuracy |
| General purpose | `mediapipe` | Balanced speed/accuracy |
| CPU-only | `ultralight` | Optimized for CPU |
| **Deepfake Detection** | | |
| High accuracy | `resnext` | Slower, more memory |
| Balanced | `resnet_inception` | Good all-around |
| Speed optimized | `efficientnet` | Faster inference |

### Memory Management

```python
# Process large batches efficiently
import gc
from mukh.face_detection import FaceDetector

detector = FaceDetector.create("mediapipe")

for batch_start in range(0, len(image_list), batch_size):
    batch = image_list[batch_start:batch_start + batch_size]
    
    for image_path in batch:
        detections = detector.detect(image_path)
        # Process results...
    
    # Clear memory between batches
    gc.collect()
```
```