# Face Detection API

The face detection module provides various models for detecting faces in images.

## Main Interface

::: mukh.face_detection.FaceDetector
    options:
      show_source: false
      heading_level: 3

## Available Models

### MediaPipe Model

::: mukh.face_detection.models.mediapipe
    options:
      show_source: false
      heading_level: 4

### BlazeFace Model

::: mukh.face_detection.models.blazeface
    options:
      show_source: false
      heading_level: 4

### UltraLight Model

::: mukh.face_detection.models.ultralight
    options:
      show_source: false
      heading_level: 4

## Usage Examples

### Basic Face Detection

```python
from mukh.face_detection import FaceDetector

# Create detector
detector = FaceDetector.create("mediapipe")

# Detect faces
detections = detector.detect(
    image_path="path/to/image.jpg",
    save_csv=True,
    csv_path="output/detections.csv",
    save_annotated=True,
    output_folder="output/annotated"
)

# Process results
for i, detection in enumerate(detections):
    print(f"Face {i+1}:")
    print(f"  Bounding box: {detection.bbox}")
    print(f"  Confidence: {detection.confidence:.3f}")
    if detection.landmarks:
        print(f"  Landmarks: {len(detection.landmarks)} points")
```

### Model Comparison

```python
from mukh.face_detection import FaceDetector

models = ["blazeface", "mediapipe", "ultralight"]
image_path = "test_image.jpg"

for model_name in models:
    detector = FaceDetector.create(model_name)
    detections = detector.detect(image_path)
    print(f"{model_name}: {len(detections)} faces detected")
```

### Custom Configuration

```python
from mukh.face_detection import FaceDetector

# Create detector with custom parameters
detector = FaceDetector.create(
    model_name="mediapipe",
    confidence_threshold=0.7,
    max_faces=5,
    min_detection_size=50
)

# Detect with custom settings
detections = detector.detect(
    image_path="image.jpg",
    draw_confidence=True,
    draw_landmarks=True,
    annotation_color=(0, 255, 0),  # Green
    line_thickness=2
)
```

### Batch Processing

```python
import os
from mukh.face_detection import FaceDetector

detector = FaceDetector.create("mediapipe")

# Process all images in a folder
input_folder = "input_images"
output_folder = "output_detections"

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)
        
        detections = detector.detect(
            image_path=image_path,
            save_csv=True,
            csv_path=os.path.join(output_folder, f"{filename}_detections.csv"),
            save_annotated=True,
            output_folder=output_folder
        )
        
        print(f"{filename}: {len(detections)} faces")
```

## Model Characteristics

| Model | Speed | Accuracy | Memory | Best Use Case |
|-------|-------|----------|---------|---------------|
| BlazeFace | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Mobile, real-time |
| MediaPipe | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | General purpose |
| UltraLight | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | CPU-only, edge devices |

## Error Handling

```python
from mukh.face_detection import FaceDetector
import os

def safe_face_detection(image_path, model_name="mediapipe"):
    try:
        # Validate input
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Create detector
        detector = FaceDetector.create(model_name)
        
        # Perform detection
        detections = detector.detect(image_path)
        
        return detections
        
    except Exception as e:
        print(f"Error in face detection: {e}")
        return []

# Usage
detections = safe_face_detection("image.jpg")
if detections:
    print(f"Successfully detected {len(detections)} faces")
else:
    print("No faces detected or error occurred")
``` 