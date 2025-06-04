# Quick Start Guide

This guide will help you get up and running with Mukh in just a few minutes.

## Basic Face Detection

Let's start with the most common use case - detecting faces in an image:

```python
from mukh.face_detection import FaceDetector

# Initialize the detector with your preferred model
detector = FaceDetector.create("mediapipe")  # Options: "blazeface", "mediapipe", "ultralight"

# Detect faces in an image
detections = detector.detect(
    image_path="path/to/your/image.jpg",
    save_csv=True,                                      # Save results to CSV
    csv_path="output/detections.csv",                   # CSV output path
    save_annotated=True,                                # Save annotated image
    output_folder="output/annotated"                    # Annotated image output folder
)

# Print detection results
print(f"Found {len(detections)} faces")
for i, detection in enumerate(detections):
    print(f"Face {i+1}: {detection}")
```

## Face Reenactment

Create realistic face reenactment videos:

```python
from mukh.face_reenactment import FaceReenactor

# Initialize the reenactor
reenactor = FaceReenactor.create("tps")  # Currently available: "tps"

# Generate reenactment video
result_path = reenactor.reenact_from_video(
    source_path="assets/images/source_face.jpg",        # Source face image
    driving_video_path="assets/videos/driving_video.mp4", # Driving video
    output_path="output/reenactment",                   # Output directory
    save_comparison=True,                               # Save side-by-side comparison
    resize_to_image_resolution=False                    # Keep original resolution
)

print(f"Reenactment video saved to: {result_path}")
```

## Working with Multiple Models

Compare results across different models:

```python
from mukh.face_detection import FaceDetector

# Available models for face detection
models = ["blazeface", "mediapipe", "ultralight"]

for model_name in models:
    print(f"\n--- Testing {model_name} ---")
    
    # Create detector for this model
    detector = FaceDetector.create(model_name)
    
    # Detect faces
    detections = detector.detect(
        image_path="test_image.jpg",
        save_csv=True,
        csv_path=f"output/{model_name}_detections.csv",
        save_annotated=True,
        output_folder=f"output/{model_name}_annotated"
    )
    
    print(f"Model: {model_name}, Faces detected: {len(detections)}")
```

## Batch Processing

Process multiple images at once:

```python
import os
from mukh.face_detection import FaceDetector

# Initialize detector
detector = FaceDetector.create("mediapipe")

# Process all images in a directory
image_directory = "path/to/your/images"
output_directory = "output/batch_processing"

for filename in os.listdir(image_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_directory, filename)
        
        # Create output subdirectory for this image
        image_output = os.path.join(output_directory, os.path.splitext(filename)[0])
        os.makedirs(image_output, exist_ok=True)
        
        # Detect faces
        detections = detector.detect(
            image_path=image_path,
            save_csv=True,
            csv_path=os.path.join(image_output, "detections.csv"),
            save_annotated=True,
            output_folder=image_output
        )
        
        print(f"{filename}: {len(detections)} faces detected")
```

## Understanding Detection Results

Each detection contains the following information:

```python
# Example detection object structure
for detection in detections:
    print(f"Bounding box: {detection.bbox}")      # [x, y, width, height]
    print(f"Confidence: {detection.confidence}")  # Detection confidence score
    print(f"Landmarks: {detection.landmarks}")    # Facial landmarks (if available)
```

## Model Selection Guidelines

Choose the right model for your use case:

=== "MediaPipe"
    - **Best for**: General purpose, real-time applications
    - **Pros**: Good balance of speed and accuracy
    - **Cons**: Requires MediaPipe dependencies

=== "BlazeFace"
    - **Best for**: Mobile applications, lightweight deployments
    - **Pros**: Very fast, lightweight
    - **Cons**: Slightly lower accuracy on challenging images

=== "UltraLight"
    - **Best for**: CPU-only environments, edge devices
    - **Pros**: Minimal dependencies, fast on CPU
    - **Cons**: Lower accuracy compared to other models

## Next Steps

Now that you've learned the basics, explore more advanced features:

- [Face Detection Examples](../examples/face-detection.md) - Detailed face detection scenarios
- [Face Reenactment Examples](../examples/face-reenactment.md) - Advanced reenactment techniques  
- [Deepfake Detection Examples](../examples/deepfake-detection.md) - Detect synthetic content
- [API Reference](../api/core.md) - Complete API documentation

## Common Patterns

### Error Handling

```python
try:
    detector = FaceDetector.create("mediapipe")
    detections = detector.detect("image.jpg")
except Exception as e:
    print(f"Error during detection: {e}")
```

### Configuration

```python
# Configure detection parameters
detector = FaceDetector.create(
    model_name="mediapipe",
    confidence_threshold=0.5,    # Minimum confidence for detection
    max_faces=10                 # Maximum number of faces to detect
)
``` 