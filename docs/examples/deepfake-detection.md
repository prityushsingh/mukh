# Deepfake Detection Examples

This page demonstrates how to use Mukh's deepfake detection capabilities to identify synthetic content in images and videos.

## Basic Deepfake Detection

The simplest way to detect deepfakes in an image:

```python
"""
Basic example showing how to detect deepfakes in media files.
"""
from mukh.deepfake_detection import DeepfakeDetector

# Create detector
detector = DeepfakeDetector(
    model_name="resnet_inception",  # Available models: "resnet_inception", "resnext", "efficientnet"
    confidence_threshold=0.5,       # Confidence threshold for deepfake detection (0.0 to 1.0)
)

# Detect deepfakes in an image
detections = detector.detect(
    media_path="assets/images/img1.jpg",                        # Path to the media file
    save_csv=True,                                              # Save the detections to a CSV file
    csv_path="output/resnet_inception/deepfake_detections.csv", # Path to save the CSV file
    save_annotated=True,                                        # Save the annotated media
    output_folder="output/resnet_inception",                    # Path to save the annotated media
)

# Process results
if hasattr(detections, 'is_deepfake'):  # Single image result
    print(f"Result: {'Deepfake' if detections.is_deepfake else 'Real'}")
    print(f"Confidence: {detections.confidence:.3f}")
else:  # Video results (list of frame detections)
    print(f"Analyzed {len(detections)} frames")
    deepfake_count = sum(1 for det in detections if det.is_deepfake)
    print(f"Deepfake frames: {deepfake_count}/{len(detections)}")
```

## Video Analysis

Analyze video files for deepfake content:

```python
"""
Analyze video files for deepfake detection.
"""
from mukh.deepfake_detection import DeepfakeDetector

# Create detector with custom configuration
detector = DeepfakeDetector(
    model_name="efficientnet",
    confidence_threshold=0.6,  # Higher threshold for more confident detections
)

# Analyze video file
video_path = "assets/videos/test_video.mp4"
detections = detector.detect(
    media_path=video_path,
    save_csv=True,
    csv_path="output/efficientnet/video_analysis.csv",
    save_annotated=True,
    output_folder="output/efficientnet",
    num_frames=20,  # Number of equally spaced frames to analyze
)

# Analyze results
if isinstance(detections, list):
    total_frames = len(detections)
    deepfake_frames = [det for det in detections if det.is_deepfake]
    real_frames = [det for det in detections if not det.is_deepfake]
    
    print(f"Video Analysis Results:")
    print(f"  Total frames analyzed: {total_frames}")
    print(f"  Deepfake frames: {len(deepfake_frames)} ({len(deepfake_frames)/total_frames*100:.1f}%)")
    print(f"  Real frames: {len(real_frames)} ({len(real_frames)/total_frames*100:.1f}%)")
    
    # Calculate average confidence for each category
    if deepfake_frames:
        avg_deepfake_conf = sum(det.confidence for det in deepfake_frames) / len(deepfake_frames)
        print(f"  Average deepfake confidence: {avg_deepfake_conf:.3f}")
    
    if real_frames:
        avg_real_conf = sum(det.confidence for det in real_frames) / len(real_frames)
        print(f"  Average real confidence: {avg_real_conf:.3f}")
    
    # Overall video classification
    deepfake_ratio = len(deepfake_frames) / total_frames
    if deepfake_ratio > 0.5:
        print(f"  Overall classification: DEEPFAKE (ratio: {deepfake_ratio:.2f})")
    else:
        print(f"  Overall classification: REAL (ratio: {1-deepfake_ratio:.2f})")
```

## Model Comparison

Compare different deepfake detection models:

```python
"""
Compare performance across different deepfake detection models.
"""
from mukh.deepfake_detection import DeepfakeDetector
import time

# Available models
models = ["resnet_inception", "resnext", "efficientnet"]
test_media = [
    "assets/images/real_face.jpg",
    "assets/images/synthetic_face.jpg",
    "assets/videos/test_video.mp4"
]

results = {}

for model_name in models:
    print(f"\n--- Testing {model_name} ---")
    
    # Create detector
    detector = DeepfakeDetector(
        model_name=model_name,
        confidence_threshold=0.5
    )
    
    model_results = []
    
    for media_path in test_media:
        print(f"  Processing: {media_path}")
        
        # Time the detection
        start_time = time.time()
        detections = detector.detect(
            media_path=media_path,
            save_csv=False,
            save_annotated=False,
            num_frames=10  # For videos
        )
        end_time = time.time()
        
        # Process results
        if isinstance(detections, list):  # Video
            deepfake_count = sum(1 for det in detections if det.is_deepfake)
            avg_confidence = sum(det.confidence for det in detections) / len(detections)
            result = {
                'media': media_path,
                'type': 'video',
                'frames_analyzed': len(detections),
                'deepfake_frames': deepfake_count,
                'avg_confidence': avg_confidence,
                'processing_time': end_time - start_time
            }
        else:  # Image
            result = {
                'media': media_path,
                'type': 'image',
                'is_deepfake': detections.is_deepfake,
                'confidence': detections.confidence,
                'processing_time': end_time - start_time
            }
        
        model_results.append(result)
    
    results[model_name] = model_results

# Summary comparison
print("\n--- Model Comparison Summary ---")
for model, model_results in results.items():
    total_time = sum(r['processing_time'] for r in model_results)
    print(f"\n{model}:")
    print(f"  Total processing time: {total_time:.3f}s")
    
    for result in model_results:
        if result['type'] == 'image':
            print(f"  {result['media']}: {'Deepfake' if result['is_deepfake'] else 'Real'} "
                  f"(conf: {result['confidence']:.3f}, time: {result['processing_time']:.3f}s)")
        else:
            print(f"  {result['media']}: {result['deepfake_frames']}/{result['frames_analyzed']} deepfake frames "
                  f"(avg conf: {result['avg_confidence']:.3f}, time: {result['processing_time']:.3f}s)")
```

## Batch Processing

Process multiple files efficiently:

```python
"""
Batch deepfake detection for multiple files.
"""
import os
from mukh.deepfake_detection import DeepfakeDetector

def batch_deepfake_detection(input_folder, output_folder, model_name="resnet_inception"):
    """
    Process all media files in a folder for deepfake detection.
    
    Args:
        input_folder (str): Folder containing media files
        output_folder (str): Output folder for results
        model_name (str): Model to use for detection
    """
    # Create detector
    detector = DeepfakeDetector(
        model_name=model_name,
        confidence_threshold=0.5
    )
    
    # Supported file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    results = []
    
    # Process all files in input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext in image_extensions or file_ext in video_extensions:
            print(f"Processing: {filename}")
            
            try:
                # Create output subfolder for this file
                file_output_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
                os.makedirs(file_output_folder, exist_ok=True)
                
                # Detect deepfakes
                detections = detector.detect(
                    media_path=file_path,
                    save_csv=True,
                    csv_path=os.path.join(file_output_folder, "detections.csv"),
                    save_annotated=True,
                    output_folder=file_output_folder,
                    num_frames=15 if file_ext in video_extensions else None
                )
                
                # Store results
                if isinstance(detections, list):  # Video
                    deepfake_count = sum(1 for det in detections if det.is_deepfake)
                    results.append({
                        'filename': filename,
                        'type': 'video',
                        'total_frames': len(detections),
                        'deepfake_frames': deepfake_count,
                        'deepfake_ratio': deepfake_count / len(detections),
                        'status': 'success'
                    })
                    print(f"  ✓ Video: {deepfake_count}/{len(detections)} deepfake frames")
                else:  # Image
                    results.append({
                        'filename': filename,
                        'type': 'image',
                        'is_deepfake': detections.is_deepfake,
                        'confidence': detections.confidence,
                        'status': 'success'
                    })
                    print(f"  ✓ Image: {'Deepfake' if detections.is_deepfake else 'Real'} "
                          f"(confidence: {detections.confidence:.3f})")
                
            except Exception as e:
                print(f"  ✗ Error processing {filename}: {str(e)}")
                results.append({
                    'filename': filename,
                    'status': 'error',
                    'error': str(e)
                })
    
    return results

# Usage
input_folder = "assets/test_media"
output_folder = "output/batch_deepfake_detection"

print("Starting batch deepfake detection...")
results = batch_deepfake_detection(input_folder, output_folder)

# Summary
successful = [r for r in results if r['status'] == 'success']
failed = [r for r in results if r['status'] == 'error']

print(f"\nBatch Processing Summary:")
print(f"  Total files processed: {len(results)}")
print(f"  Successful: {len(successful)}")
print(f"  Failed: {len(failed)}")

if successful:
    images = [r for r in successful if r['type'] == 'image']
    videos = [r for r in successful if r['type'] == 'video']
    
    if images:
        deepfake_images = [r for r in images if r['is_deepfake']]
        print(f"  Images: {len(deepfake_images)}/{len(images)} detected as deepfakes")
    
    if videos:
        high_deepfake_videos = [r for r in videos if r['deepfake_ratio'] > 0.5]
        print(f"  Videos: {len(high_deepfake_videos)}/{len(videos)} classified as deepfakes")
```

## Advanced Configuration and Thresholds

Fine-tune detection parameters:

```python
"""
Advanced deepfake detection with custom thresholds and analysis.
"""
from mukh.deepfake_detection import DeepfakeDetector
import numpy as np

def advanced_deepfake_analysis(media_path, models=None, thresholds=None):
    """
    Perform advanced deepfake analysis with multiple models and thresholds.
    
    Args:
        media_path (str): Path to media file
        models (list): List of models to use
        thresholds (list): List of confidence thresholds to test
    
    Returns:
        dict: Comprehensive analysis results
    """
    if models is None:
        models = ["resnet_inception", "resnext", "efficientnet"]
    
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7, 0.9]
    
    analysis_results = {}
    
    for model_name in models:
        print(f"Analyzing with {model_name}...")
        model_results = {}
        
        for threshold in thresholds:
            # Create detector with specific threshold
            detector = DeepfakeDetector(
                model_name=model_name,
                confidence_threshold=threshold
            )
            
            # Perform detection
            detections = detector.detect(
                media_path=media_path,
                save_csv=False,
                save_annotated=False,
                num_frames=20
            )
            
            # Analyze results
            if isinstance(detections, list):  # Video
                raw_confidences = [det.confidence for det in detections]
                deepfake_count = sum(1 for det in detections if det.is_deepfake)
                
                result = {
                    'total_frames': len(detections),
                    'deepfake_frames': deepfake_count,
                    'deepfake_ratio': deepfake_count / len(detections),
                    'avg_confidence': np.mean(raw_confidences),
                    'std_confidence': np.std(raw_confidences),
                    'min_confidence': np.min(raw_confidences),
                    'max_confidence': np.max(raw_confidences),
                    'confidence_distribution': np.histogram(raw_confidences, bins=10)[0].tolist()
                }
            else:  # Image
                result = {
                    'is_deepfake': detections.is_deepfake,
                    'confidence': detections.confidence
                }
            
            model_results[f'threshold_{threshold}'] = result
        
        analysis_results[model_name] = model_results
    
    return analysis_results

# Perform comprehensive analysis
media_file = "assets/videos/suspicious_video.mp4"
results = advanced_deepfake_analysis(media_file)

# Print detailed analysis
print(f"\nComprehensive Deepfake Analysis for: {media_file}")
print("=" * 60)

for model, model_results in results.items():
    print(f"\nModel: {model}")
    print("-" * 40)
    
    for threshold_key, result in model_results.items():
        threshold = threshold_key.split('_')[1]
        print(f"\nThreshold: {threshold}")
        
        if 'total_frames' in result:  # Video results
            print(f"  Deepfake frames: {result['deepfake_frames']}/{result['total_frames']} "
                  f"({result['deepfake_ratio']:.1%})")
            print(f"  Avg confidence: {result['avg_confidence']:.3f} ± {result['std_confidence']:.3f}")
            print(f"  Confidence range: {result['min_confidence']:.3f} - {result['max_confidence']:.3f}")
        else:  # Image results
            print(f"  Classification: {'Deepfake' if result['is_deepfake'] else 'Real'}")
            print(f"  Confidence: {result['confidence']:.3f}")

# Model agreement analysis
print("\n" + "=" * 60)
print("Model Agreement Analysis")
print("=" * 60)

# For videos, analyze agreement at different thresholds
threshold = 0.5
agreements = {}

for model in results.keys():
    model_result = results[model][f'threshold_{threshold}']
    if 'deepfake_ratio' in model_result:
        agreements[model] = model_result['deepfake_ratio'] > 0.5  # Consider as deepfake if >50% frames
    else:
        agreements[model] = model_result['is_deepfake']

agreement_count = sum(agreements.values())
total_models = len(agreements)

print(f"\nAt threshold {threshold}:")
for model, is_deepfake in agreements.items():
    print(f"  {model}: {'Deepfake' if is_deepfake else 'Real'}")

print(f"\nConsensus: {agreement_count}/{total_models} models agree")
if agreement_count == total_models:
    consensus = "UNANIMOUS - " + ("Deepfake" if list(agreements.values())[0] else "Real")
elif agreement_count >= total_models * 0.66:
    majority_vote = agreement_count > total_models / 2
    consensus = f"MAJORITY - {'Deepfake' if majority_vote else 'Real'}"
else:
    consensus = "NO CLEAR CONSENSUS"

print(f"Final assessment: {consensus}")
```

## Real-time Detection

Implement real-time deepfake detection:

```python
"""
Real-time deepfake detection from webcam or video stream.
"""
import cv2
from mukh.deepfake_detection import DeepfakeDetector
import time

def real_time_deepfake_detection(source=0, model_name="efficientnet"):
    """
    Perform real-time deepfake detection from video source.
    
    Args:
        source: Video source (0 for webcam, or path to video file)
        model_name (str): Model to use for detection
    """
    # Initialize detector
    detector = DeepfakeDetector(
        model_name=model_name,
        confidence_threshold=0.5
    )
    
    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source {source}")
        return
    
    # Detection parameters
    detection_interval = 30  # Detect every N frames for performance
    frame_count = 0
    last_detection = None
    detection_history = []
    
    print("Real-time deepfake detection started. Press 'q' to quit.")
    print("Press 's' to save current frame for analysis.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame")
                break
            
            frame_count += 1
            
            # Perform detection periodically
            if frame_count % detection_interval == 0:
                # Save frame temporarily
                temp_frame_path = "temp_realtime_frame.jpg"
                cv2.imwrite(temp_frame_path, frame)
                
                try:
                    # Perform detection
                    start_time = time.time()
                    detection = detector.detect(
                        media_path=temp_frame_path,
                        save_csv=False,
                        save_annotated=False
                    )
                    detection_time = time.time() - start_time
                    
                    last_detection = {
                        'is_deepfake': detection.is_deepfake,
                        'confidence': detection.confidence,
                        'timestamp': time.time(),
                        'detection_time': detection_time
                    }
                    
                    # Store in history (keep last 10 detections)
                    detection_history.append(last_detection)
                    if len(detection_history) > 10:
                        detection_history.pop(0)
                    
                except Exception as e:
                    print(f"Detection error: {e}")
                    last_detection = None
            
            # Display results on frame
            if last_detection:
                # Determine display color
                color = (0, 0, 255) if last_detection['is_deepfake'] else (0, 255, 0)  # Red for deepfake, green for real
                
                # Add detection info to frame
                label = f"{'DEEPFAKE' if last_detection['is_deepfake'] else 'REAL'}"
                confidence_text = f"Conf: {last_detection['confidence']:.3f}"
                time_text = f"Det: {last_detection['detection_time']:.2f}s"
                
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, confidence_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, time_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Show detection history trend
                if len(detection_history) > 1:
                    deepfake_count = sum(1 for d in detection_history if d['is_deepfake'])
                    trend_text = f"Recent: {deepfake_count}/{len(detection_history)} deepfake"
                    cv2.putText(frame, trend_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add frame info
            fps_text = f"Frame: {frame_count}"
            cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Real-time Deepfake Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and last_detection:
                # Save current frame with detection info
                timestamp = int(time.time())
                label = "deepfake" if last_detection['is_deepfake'] else "real"
                save_path = f"saved_frame_{timestamp}_{label}_{last_detection['confidence']:.3f}.jpg"
                cv2.imwrite(save_path, frame)
                print(f"Saved frame: {save_path}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Clean up temporary files
        import os
        if os.path.exists("temp_realtime_frame.jpg"):
            os.remove("temp_realtime_frame.jpg")
        
        # Print session summary
        if detection_history:
            print(f"\nSession Summary:")
            print(f"  Total detections: {len(detection_history)}")
            deepfake_detections = [d for d in detection_history if d['is_deepfake']]
            print(f"  Deepfake detections: {len(deepfake_detections)}")
            avg_conf = sum(d['confidence'] for d in detection_history) / len(detection_history)
            print(f"  Average confidence: {avg_conf:.3f}")
            avg_time = sum(d['detection_time'] for d in detection_history) / len(detection_history)
            print(f"  Average detection time: {avg_time:.3f}s")

# Usage examples
print("Choose detection source:")
print("1. Webcam (press 1)")
print("2. Video file (press 2)")

choice = input("Enter choice (1 or 2): ")

if choice == "1":
    real_time_deepfake_detection(source=0)  # Webcam
elif choice == "2":
    video_path = input("Enter video file path: ")
    real_time_deepfake_detection(source=video_path)
else:
    print("Invalid choice")
```

## Next Steps

- Learn about [Face Detection](face-detection.md) for preprocessing face regions
- Explore [Face Reenactment](face-reenactment.md) to understand synthetic content generation
- Check the [API Reference](../api/deepfake-detection.md) for detailed documentation
- Combine deepfake detection with other Mukh features for comprehensive analysis pipelines 