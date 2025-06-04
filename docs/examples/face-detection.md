# Face Detection Examples

This page provides comprehensive examples of using Mukh's face detection capabilities.

## Basic Face Detection

The simplest way to detect faces in a single image:

```python
"""
Basic example showing how to detect faces in a single image.
"""
from mukh.face_detection import FaceDetector

# Create detector with your preferred model
detector = FaceDetector.create("mediapipe")  # Options: "blazeface", "mediapipe", "ultralight"

# Detect faces
detections = detector.detect(
    image_path="assets/images/img1.jpg",                       # Path to the image
    save_csv=True,                                             # Save detections to CSV
    csv_path="output/mediapipe/detections.csv",                # CSV output path
    save_annotated=True,                                       # Save annotated image
    output_folder="output/mediapipe",                          # Output directory
)

# Process results
print(f"Found {len(detections)} faces")
for i, detection in enumerate(detections):
    print(f"Face {i+1}: {detection}")
```

## Batch Processing

Process multiple images in a folder:

```python
"""
Process all images in a directory for face detection.
"""
import os
from mukh.face_detection import FaceDetector

# Initialize detector
detector = FaceDetector.create("mediapipe")

# Process all images in the folder
images_folder = "assets/images"
for image_name in os.listdir(images_folder):
    if image_name.endswith((".jpg", ".png", ".jpeg")):
        # Get full image path
        image_path = os.path.join(images_folder, image_name)
        
        print(f"Processing: {image_name}")
        
        # Detect faces
        detections = detector.detect(
            image_path=image_path,
            save_csv=True,
            csv_path=f"output/mediapipe/detections_{image_name}.csv",
            save_annotated=True,
            output_folder="output/mediapipe",
        )
        
        print(f"  Found {len(detections)} faces")
```

## Model Comparison

Compare different face detection models:

```python
"""
Compare performance across different face detection models.
"""
from mukh.face_detection import FaceDetector
import time

# Available models
models = ["blazeface", "mediapipe", "ultralight"]
image_path = "assets/images/test_image.jpg"

results = {}

for model_name in models:
    print(f"\n--- Testing {model_name} ---")
    
    # Create detector
    detector = FaceDetector.create(model_name)
    
    # Time the detection
    start_time = time.time()
    detections = detector.detect(
        image_path=image_path,
        save_csv=True,
        csv_path=f"output/{model_name}/detections.csv",
        save_annotated=True,
        output_folder=f"output/{model_name}",
    )
    end_time = time.time()
    
    # Store results
    results[model_name] = {
        'faces_detected': len(detections),
        'processing_time': end_time - start_time
    }
    
    print(f"Faces detected: {len(detections)}")
    print(f"Processing time: {end_time - start_time:.3f} seconds")

# Summary
print("\n--- Summary ---")
for model, result in results.items():
    print(f"{model}: {result['faces_detected']} faces, {result['processing_time']:.3f}s")
```

## Advanced Configuration

Customize detection parameters:

```python
"""
Advanced face detection with custom configuration.
"""
from mukh.face_detection import FaceDetector

# Create detector with custom parameters
detector = FaceDetector.create(
    model_name="mediapipe",
    confidence_threshold=0.7,    # Higher confidence threshold
    max_faces=5,                 # Limit number of faces to detect
    min_detection_size=50        # Minimum face size in pixels
)

# Detect with custom settings
detections = detector.detect(
    image_path="assets/images/crowd.jpg",
    save_csv=True,
    csv_path="output/custom/detections.csv",
    save_annotated=True,
    output_folder="output/custom",
    draw_confidence=True,        # Show confidence scores on image
    draw_landmarks=True          # Draw facial landmarks if available
)

# Analyze detection confidence
confidences = [det.confidence for det in detections]
if confidences:
    avg_confidence = sum(confidences) / len(confidences)
    print(f"Average confidence: {avg_confidence:.3f}")
    print(f"Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
```

## Real-time Detection

Process video frames or webcam feed:

```python
"""
Real-time face detection from video or webcam.
"""
import cv2
from mukh.face_detection import FaceDetector

# Initialize detector
detector = FaceDetector.create("mediapipe")

# Open video source (0 for webcam, or path to video file)
cap = cv2.VideoCapture(0)  # Use webcam
# cap = cv2.VideoCapture("path/to/video.mp4")  # Use video file

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame temporarily for detection
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)
        
        # Detect faces
        detections = detector.detect(
            image_path=temp_path,
            save_csv=False,
            save_annotated=False
        )
        
        # Draw bounding boxes on frame
        for detection in detections:
            x, y, w, h = detection.bbox
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
            cv2.putText(frame, f"{detection.confidence:.2f}", 
                       (int(x), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Face Detection', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
finally:
    cap.release()
    cv2.destroyAllWindows()
    # Clean up temporary file
    import os
    if os.path.exists("temp_frame.jpg"):
        os.remove("temp_frame.jpg")
```

## Error Handling and Validation

Robust face detection with proper error handling:

```python
"""
Face detection with comprehensive error handling.
"""
from mukh.face_detection import FaceDetector
import os

def safe_face_detection(image_path, model_name="mediapipe"):
    """
    Perform face detection with error handling.
    
    Args:
        image_path (str): Path to the image file
        model_name (str): Model to use for detection
    
    Returns:
        list: Detection results or empty list if error
    """
    try:
        # Validate input file
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found")
            return []
        
        # Check file extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in valid_extensions:
            print(f"Error: Unsupported file format '{file_ext}'")
            return []
        
        # Create detector
        detector = FaceDetector.create(model_name)
        
        # Perform detection
        detections = detector.detect(
            image_path=image_path,
            save_csv=True,
            csv_path=f"output/{model_name}/detections.csv",
            save_annotated=True,
            output_folder=f"output/{model_name}"
        )
        
        print(f"Successfully detected {len(detections)} faces in '{image_path}'")
        return detections
        
    except Exception as e:
        print(f"Error during face detection: {str(e)}")
        return []

# Usage
image_paths = [
    "assets/images/img1.jpg",
    "assets/images/img2.jpg", 
    "non_existent_image.jpg"  # This will be handled gracefully
]

for image_path in image_paths:
    detections = safe_face_detection(image_path)
    if detections:
        for i, detection in enumerate(detections):
            print(f"  Face {i+1}: confidence={detection.confidence:.3f}")
```

## Performance Benchmarking

Benchmark different models for your use case:

```python
"""
Benchmark face detection models for performance analysis.
"""
from mukh.face_detection import FaceDetector
import time
import statistics

def benchmark_model(model_name, image_paths, num_runs=3):
    """
    Benchmark a face detection model.
    
    Args:
        model_name (str): Name of the model to benchmark
        image_paths (list): List of image paths to test
        num_runs (int): Number of runs for averaging
    
    Returns:
        dict: Benchmark results
    """
    detector = FaceDetector.create(model_name)
    times = []
    total_faces = 0
    
    for run in range(num_runs):
        run_start = time.time()
        run_faces = 0
        
        for image_path in image_paths:
            detections = detector.detect(
                image_path=image_path,
                save_csv=False,
                save_annotated=False
            )
            run_faces += len(detections)
        
        run_time = time.time() - run_start
        times.append(run_time)
        if run == 0:  # Only count faces once
            total_faces = run_faces
    
    return {
        'model': model_name,
        'avg_time': statistics.mean(times),
        'std_time': statistics.stdev(times) if len(times) > 1 else 0,
        'total_faces': total_faces,
        'images_processed': len(image_paths),
        'fps': len(image_paths) / statistics.mean(times)
    }

# Run benchmark
models = ["blazeface", "mediapipe", "ultralight"]
test_images = [
    "assets/images/img1.jpg",
    "assets/images/img2.jpg",
    # Add more test images
]

print("Benchmarking face detection models...\n")
results = []

for model in models:
    print(f"Testing {model}...")
    result = benchmark_model(model, test_images)
    results.append(result)
    
    print(f"  Average time: {result['avg_time']:.3f} ± {result['std_time']:.3f}s")
    print(f"  Total faces detected: {result['total_faces']}")
    print(f"  Processing speed: {result['fps']:.1f} images/second\n")

# Find best model for different criteria
fastest = min(results, key=lambda x: x['avg_time'])
print(f"Fastest model: {fastest['model']} ({fastest['avg_time']:.3f}s)")
```

## Output Formats

Understanding detection output formats:

```python
"""
Understanding and working with detection output formats.
"""
from mukh.face_detection import FaceDetector
import json
import pandas as pd

detector = FaceDetector.create("mediapipe")

# Detect faces
detections = detector.detect(
    image_path="assets/images/img1.jpg",
    save_csv=True,
    csv_path="output/detections.csv"
)

# Work with detection objects
for i, detection in enumerate(detections):
    print(f"Face {i+1}:")
    print(f"  Bounding box: {detection.bbox}")  # [x, y, width, height]
    print(f"  Confidence: {detection.confidence}")
    print(f"  Landmarks: {detection.landmarks}")  # If available
    print()

# Convert to different formats
detection_data = []
for detection in detections:
    detection_data.append({
        'bbox_x': detection.bbox[0],
        'bbox_y': detection.bbox[1], 
        'bbox_width': detection.bbox[2],
        'bbox_height': detection.bbox[3],
        'confidence': detection.confidence,
        'landmarks': detection.landmarks
    })

# Save as JSON
with open('output/detections.json', 'w') as f:
    json.dump(detection_data, f, indent=2)

# Save as DataFrame
df = pd.DataFrame(detection_data)
df.to_csv('output/detections_detailed.csv', index=False)
df.to_excel('output/detections.xlsx', index=False)

print(f"Saved {len(detections)} detections in multiple formats")
```

## Next Steps

- Learn about [Face Reenactment](face-reenactment.md) for video generation
- Explore [Deepfake Detection](deepfake-detection.md) for content verification
- Check the [API Reference](../api/face-detection.md) for detailed documentation 