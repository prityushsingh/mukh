# Deepfake Detection API

The deepfake detection module provides functionality for detecting deepfake content in images and videos.

## DeepfakeDetector

::: mukh.deepfake_detection.DeepfakeDetector
    options:
      show_source: false
      heading_level: 3

## Examples

### Basic Deepfake Detection

```python
from mukh.deepfake_detection import DeepfakeDetector

# Create detector
detector = DeepfakeDetector(
    model_name="resnet_inception",  # Options: resnet_inception, resnext, efficientnet
    confidence_threshold=0.5
)

# Analyze an image
detection = detector.detect(
    media_path="suspect_image.jpg",
    save_csv=True,
    csv_path="output/deepfake_results.csv",
    save_annotated=True,
    output_folder="output"
)

# Check result
if detection.is_deepfake:
    print(f"Deepfake detected with confidence: {detection.confidence:.3f}")
else:
    print(f"Real content with confidence: {detection.confidence:.3f}")
```

### Video Analysis

```python
from mukh.deepfake_detection import DeepfakeDetector

detector = DeepfakeDetector(model_name="efficientnet")

# Analyze video with frame sampling
detections = detector.detect(
    media_path="video.mp4",
    num_frames=15,  # Analyze 15 equally spaced frames
    save_csv=True,
    output_folder="video_analysis"
)

# Analyze results
deepfake_count = sum(1 for d in detections if d.is_deepfake)
total_frames = len(detections)
print(f"Deepfake frames: {deepfake_count}/{total_frames}")

# Frame-by-frame analysis
for detection in detections:
    status = "Deepfake" if detection.is_deepfake else "Real"
    print(f"Frame {detection.frame_number}: {status} (confidence: {detection.confidence:.3f})")
```

### Multi-Model Consensus

```python
from mukh.deepfake_detection import DeepfakeDetector

# Use multiple models for better accuracy
models = ["resnet_inception", "resnext", "efficientnet"]
detectors = [DeepfakeDetector(model_name=model) for model in models]

def consensus_detection(media_path: str, detectors: list):
    """Get consensus from multiple models."""
    results = []
    for detector in detectors:
        result = detector.detect(media_path=media_path)
        results.append(result)
    
    # Majority vote
    deepfake_votes = sum(1 for r in results if r.is_deepfake)
    avg_confidence = sum(r.confidence for r in results) / len(results)
    
    return {
        "is_deepfake": deepfake_votes > len(detectors) / 2,
        "consensus_confidence": avg_confidence,
        "individual_results": results
    }

# Analyze with consensus
result = consensus_detection("suspicious_video.mp4", detectors)
print(f"Consensus result: {'Deepfake' if result['is_deepfake'] else 'Real'}")
print(f"Average confidence: {result['consensus_confidence']:.3f}")
```

## Available Models

### ResNet Inception

::: mukh.deepfake_detection.models.resnet_inception
    options:
      show_source: false
      heading_level: 4

### ResNeXt

::: mukh.deepfake_detection.models.resnext
    options:
      show_source: false
      heading_level: 4

### EfficientNet

::: mukh.deepfake_detection.models.efficientnet
    options:
      show_source: false
      heading_level: 4

## Model Characteristics

| Model | Architecture | Accuracy | Speed | Memory | Best Use Case |
|-------|-------------|----------|-------|---------|---------------|
| ResNet Inception | ResNet + Inception | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Balanced performance |
| ResNeXt | ResNeXt | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | High accuracy |
| EfficientNet | EfficientNet | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Efficiency optimized |

## Detection Confidence Interpretation

### Confidence Thresholds

```python
def interpret_confidence(confidence, threshold=0.5):
    """Interpret detection confidence score."""
    if confidence >= 0.9:
        return "Very High Confidence"
    elif confidence >= 0.7:
        return "High Confidence"
    elif confidence >= threshold:
        return "Medium Confidence"
    elif confidence >= 0.3:
        return "Low Confidence"
    else:
        return "Very Low Confidence"

# Usage
detector = DeepfakeDetector("efficientnet", 0.5)
result = detector.detect("image.jpg")

confidence_level = interpret_confidence(result.confidence)
status = "DEEPFAKE" if result.is_deepfake else "REAL"

print(f"Detection: {status}")
print(f"Confidence: {result.confidence:.3f} ({confidence_level})")
```

### Threshold Sensitivity Analysis

```python
from mukh.deepfake_detection import DeepfakeDetector

def analyze_threshold_sensitivity(image_path, thresholds=[0.3, 0.5, 0.7, 0.9]):
    """Analyze how detection changes with different thresholds."""
    results = {}
    
    for threshold in thresholds:
        detector = DeepfakeDetector("resnet_inception", threshold)
        result = detector.detect(image_path)
        
        results[threshold] = {
            'is_deepfake': result.is_deepfake,
            'confidence': result.confidence
        }
    
    print(f"Threshold sensitivity analysis for: {image_path}")
    print("Threshold | Classification | Raw Confidence")
    print("-" * 45)
    
    for threshold, result in results.items():
        classification = "DEEPFAKE" if result['is_deepfake'] else "REAL"
        print(f"{threshold:8.1f} | {classification:13s} | {result['confidence']:.3f}")

# Usage
analyze_threshold_sensitivity("test_image.jpg")
```

## Real-time Detection

```python
import cv2
from mukh.deepfake_detection import DeepfakeDetector
import time

def real_time_detection(source=0):
    """Real-time deepfake detection from webcam or video."""
    detector = DeepfakeDetector("efficientnet", 0.5)
    cap = cv2.VideoCapture(source)
    
    detection_interval = 30  # Detect every 30 frames
    frame_count = 0
    last_detection = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Perform detection periodically
        if frame_count % detection_interval == 0:
            # Save frame temporarily
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Detect
            start_time = time.time()
            detection = detector.detect(temp_path)
            detection_time = time.time() - start_time
            
            last_detection = {
                'is_deepfake': detection.is_deepfake,
                'confidence': detection.confidence,
                'detection_time': detection_time
            }
        
        # Display results
        if last_detection:
            color = (0, 0, 255) if last_detection['is_deepfake'] else (0, 255, 0)
            label = "DEEPFAKE" if last_detection['is_deepfake'] else "REAL"
            confidence = last_detection['confidence']
            
            cv2.putText(frame, f"{label}: {confidence:.3f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow('Deepfake Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Usage
real_time_detection(0)  # Webcam
# real_time_detection("video.mp4")  # Video file
```

## Advanced Analysis

### Detection Uncertainty

```python
from mukh.deepfake_detection import DeepfakeDetector
import numpy as np

def calculate_uncertainty(detections, model_name="resnet_inception"):
    """Calculate detection uncertainty metrics."""
    if not isinstance(detections, list):
        detections = [detections]
    
    confidences = [det.confidence for det in detections]
    predictions = [det.is_deepfake for det in detections]
    
    # Statistical measures
    mean_confidence = np.mean(confidences)
    std_confidence = np.std(confidences)
    min_confidence = np.min(confidences)
    max_confidence = np.max(confidences)
    
    # Consistency measures
    deepfake_ratio = sum(predictions) / len(predictions)
    consistency = abs(deepfake_ratio - 0.5) * 2  # 0 = uncertain, 1 = consistent
    
    # Uncertainty score (lower = more uncertain)
    uncertainty = 1 - (consistency * (1 - std_confidence))
    
    return {
        'mean_confidence': mean_confidence,
        'std_confidence': std_confidence,
        'confidence_range': (min_confidence, max_confidence),
        'deepfake_ratio': deepfake_ratio,
        'consistency': consistency,
        'uncertainty': uncertainty,
        'interpretation': 'High uncertainty' if uncertainty > 0.7 else 
                        'Medium uncertainty' if uncertainty > 0.4 else 
                        'Low uncertainty'
    }

# Usage with video
detector = DeepfakeDetector("efficientnet", 0.5)
detections = detector.detect("video.mp4", num_frames=30)
uncertainty = calculate_uncertainty(detections)

print(f"Uncertainty Analysis:")
print(f"  Mean confidence: {uncertainty['mean_confidence']:.3f}")
print(f"  Std deviation: {uncertainty['std_confidence']:.3f}")
print(f"  Deepfake ratio: {uncertainty['deepfake_ratio']:.2%}")
print(f"  Consistency: {uncertainty['consistency']:.3f}")
print(f"  Uncertainty: {uncertainty['uncertainty']:.3f} ({uncertainty['interpretation']})")
```

## Error Handling

```python
from mukh.deepfake_detection import DeepfakeDetector
import os

def safe_deepfake_detection(media_path, model_name="resnet_inception"):
    """Perform deepfake detection with comprehensive error handling."""
    try:
        # Validate input
        if not os.path.exists(media_path):
            raise FileNotFoundError(f"Media file not found: {media_path}")
        
        # Check file size
        file_size = os.path.getsize(media_path)
        if file_size == 0:
            raise ValueError("File is empty")
        
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            print("Warning: Large file may take significant time to process")
        
        # Create detector
        detector = DeepfakeDetector(model_name, confidence_threshold=0.5)
        
        # Determine if image or video
        is_video = media_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        
        # Perform detection
        if is_video:
            detections = detector.detect(media_path, num_frames=15)
            return {
                'type': 'video',
                'detections': detections,
                'summary': {
                    'total_frames': len(detections),
                    'deepfake_frames': sum(1 for det in detections if det.is_deepfake),
                    'avg_confidence': sum(det.confidence for det in detections) / len(detections)
                }
            }
        else:
            detection = detector.detect(media_path)
            return {
                'type': 'image',
                'detection': detection,
                'summary': {
                    'is_deepfake': detection.is_deepfake,
                    'confidence': detection.confidence
                }
            }
        
    except Exception as e:
        return {
            'error': str(e),
            'type': 'error'
        }

# Usage
result = safe_deepfake_detection("test_media.jpg")

if result['type'] == 'error':
    print(f"Error: {result['error']}")
elif result['type'] == 'image':
    summary = result['summary']
    status = "DEEPFAKE" if summary['is_deepfake'] else "REAL"
    print(f"Image analysis: {status} (confidence: {summary['confidence']:.3f})")
elif result['type'] == 'video':
    summary = result['summary']
    ratio = summary['deepfake_frames'] / summary['total_frames']
    print(f"Video analysis: {summary['deepfake_frames']}/{summary['total_frames']} "
          f"deepfake frames ({ratio:.1%})")
``` 