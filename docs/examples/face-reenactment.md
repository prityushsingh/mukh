# Face Reenactment Examples

This page demonstrates how to use Mukh's face reenactment capabilities to generate realistic face animations.

## Basic Face Reenactment

The simplest way to create a face reenactment video:

```python
"""
Basic example showing how to reenact a face from a source image using a driving video.
"""
from mukh.reenactment import FaceReenactor

# Initialize the reenactor
reenactor = FaceReenactor.create("tps")  # Currently available: "tps"

# Generate reenactment video
result_path = reenactor.reenact_from_video(
    source_path="assets/images/img1.jpg",                       # Source face image
    driving_video_path="assets/videos/video.mp4",               # Driving video
    output_path="output/tps",                                   # Output directory
    save_comparison=True,                                       # Save side-by-side comparison
    resize_to_image_resolution=False,                           # Keep original resolution
)

print(f"Reenactment video saved to: {result_path}")
```

## Batch Reenactment

Process multiple source images with the same driving video:

```python
"""
Batch reenactment for multiple source images.
"""
import os
from mukh.reenactment import FaceReenactor

# Initialize reenactor
reenactor = FaceReenactor.create("tps")

# Configuration
driving_video_path = "assets/videos/driving_video.mp4"
source_images_folder = "assets/images"
output_base_folder = "output/batch_reenactment"

# Process all images in the source folder
for image_name in os.listdir(source_images_folder):
    if image_name.endswith((".jpg", ".png", ".jpeg")):
        # Get paths
        source_path = os.path.join(source_images_folder, image_name)
        image_output_folder = os.path.join(output_base_folder, os.path.splitext(image_name)[0])
        
        print(f"Processing: {image_name}")
        
        try:
            # Generate reenactment
            result_path = reenactor.reenact_from_video(
                source_path=source_path,
                driving_video_path=driving_video_path,
                output_path=image_output_folder,
                save_comparison=True,
                resize_to_image_resolution=False
            )
            
            print(f"  ✓ Saved to: {result_path}")
            
        except Exception as e:
            print(f"  ✗ Error processing {image_name}: {str(e)}")
```

## Custom Configuration

Advanced reenactment with custom parameters:

```python
"""
Advanced face reenactment with custom configuration.
"""
from mukh.reenactment import FaceReenactor

# Create reenactor with custom settings
reenactor = FaceReenactor.create(
    model_name="tps",
    num_kp=15,                    # Number of keypoints
    num_channels=3,               # Number of channels
    estimate_jacobian=True,       # Estimate jacobian for better quality
    temperature=0.1,              # Temperature for keypoint detection
    block_expansion=32,           # Block expansion factor
    max_features=1024,            # Maximum features to extract
    scale_factor=0.25,            # Scale factor for feature pyramid
    num_down_blocks=2,            # Number of downsampling blocks
    num_bottleneck_blocks=6,      # Number of bottleneck blocks
    estimate_occlusion_map=True,  # Estimate occlusion map
    dense_motion_params={
        'block_expansion': 64,
        'num_blocks': 5,
        'max_features': 1024,
        'num_down_blocks': 2,
        'num_bottleneck_blocks': 6,
        'estimate_occlusion_map': True,
        'scale_factor': 0.25,
        'kp_variance': 0.01
    }
)

# Generate high-quality reenactment
result_path = reenactor.reenact_from_video(
    source_path="assets/images/high_res_face.jpg",
    driving_video_path="assets/videos/expressive_video.mp4",
    output_path="output/high_quality",
    save_comparison=True,
    resize_to_image_resolution=True,           # Match source image resolution
    fps=30,                                    # Output video FPS
    quality='high',                            # Output quality
    codec='h264'                               # Video codec
)

print(f"High-quality reenactment saved to: {result_path}")
```

## Multi-driving Video Reenactment

Use multiple driving videos with the same source:

```python
"""
Generate reenactments using multiple driving videos.
"""
from mukh.reenactment import FaceReenactor
import os

# Initialize reenactor
reenactor = FaceReenactor.create("tps")

# Configuration
source_image = "assets/images/portrait.jpg"
driving_videos_folder = "assets/videos/driving_videos"
output_folder = "output/multi_driving"

# Process all driving videos
for video_name in os.listdir(driving_videos_folder):
    if video_name.endswith((".mp4", ".avi", ".mov")):
        # Get paths
        driving_video_path = os.path.join(driving_videos_folder, video_name)
        video_output_folder = os.path.join(output_folder, os.path.splitext(video_name)[0])
        
        print(f"Processing with driving video: {video_name}")
        
        # Generate reenactment
        result_path = reenactor.reenact_from_video(
            source_path=source_image,
            driving_video_path=driving_video_path,
            output_path=video_output_folder,
            save_comparison=True,
            resize_to_image_resolution=False
        )
        
        print(f"  Result saved to: {result_path}")
```

## Video Analysis and Preprocessing

Analyze and preprocess videos for better reenactment:

```python
"""
Analyze and preprocess videos for optimal reenactment results.
"""
import cv2
from mukh.reenactment import FaceReenactor
from mukh.face_detection import FaceDetector

def analyze_video(video_path):
    """
    Analyze a video for face reenactment suitability.
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        dict: Analysis results
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    # Initialize face detector for analysis
    detector = FaceDetector.create("mediapipe")
    
    # Analyze a sample of frames
    sample_frames = 10
    frame_step = max(1, frame_count // sample_frames)
    face_detections = []
    
    for i in range(0, frame_count, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame temporarily for detection
        temp_frame_path = "temp_analysis_frame.jpg"
        cv2.imwrite(temp_frame_path, frame)
        
        # Detect faces
        detections = detector.detect(
            image_path=temp_frame_path,
            save_csv=False,
            save_annotated=False
        )
        
        face_detections.append(len(detections))
    
    cap.release()
    
    # Clean up
    import os
    if os.path.exists("temp_analysis_frame.jpg"):
        os.remove("temp_analysis_frame.jpg")
    
    # Calculate statistics
    avg_faces = sum(face_detections) / len(face_detections) if face_detections else 0
    face_consistency = len([x for x in face_detections if x == 1]) / len(face_detections) if face_detections else 0
    
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'duration': duration,
        'frame_count': frame_count,
        'avg_faces_per_frame': avg_faces,
        'face_consistency': face_consistency,
        'suitable_for_reenactment': face_consistency > 0.8 and avg_faces >= 0.9
    }

def preprocess_video(input_path, output_path, target_fps=25, target_resolution=(256, 256)):
    """
    Preprocess a video for better reenactment results.
    
    Args:
        input_path (str): Input video path
        output_path (str): Output video path
        target_fps (int): Target FPS
        target_resolution (tuple): Target resolution (width, height)
    """
    cap = cv2.VideoCapture(input_path)
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, target_resolution)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        frame = cv2.resize(frame, target_resolution, interpolation=cv2.INTER_LANCZOS4)
        
        # Optional: Apply preprocessing filters
        # frame = cv2.bilateralFilter(frame, 9, 75, 75)  # Noise reduction
        # frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)  # Contrast enhancement
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    print(f"Preprocessed video saved to: {output_path}")
    print(f"Processed {frame_count} frames")

# Usage example
driving_video = "assets/videos/raw_driving_video.mp4"

# Analyze the video
analysis = analyze_video(driving_video)
print("Video Analysis Results:")
for key, value in analysis.items():
    print(f"  {key}: {value}")

if analysis['suitable_for_reenactment']:
    print("✓ Video is suitable for reenactment")
else:
    print("⚠ Video may not be optimal for reenactment")
    print("  Consider using a video with:")
    print("  - Consistent single face presence")
    print("  - Good lighting conditions")
    print("  - Minimal occlusions")

# Preprocess if needed
if analysis['fps'] > 30 or analysis['width'] > 512:
    preprocessed_video = "assets/videos/preprocessed_driving_video.mp4"
    preprocess_video(driving_video, preprocessed_video)
    driving_video = preprocessed_video

# Generate reenactment
reenactor = FaceReenactor.create("tps")
result_path = reenactor.reenact_from_video(
    source_path="assets/images/source.jpg",
    driving_video_path=driving_video,
    output_path="output/preprocessed_reenactment",
    save_comparison=True
)
```

## Quality Assessment

Evaluate reenactment quality:

```python
"""
Assess the quality of generated reenactment videos.
"""
import cv2
import numpy as np
from mukh.reenactment import FaceReenactor
from mukh.face_detection import FaceDetector

def assess_reenactment_quality(original_video_path, reenacted_video_path):
    """
    Assess the quality of a reenactment video.
    
    Args:
        original_video_path (str): Path to original driving video
        reenacted_video_path (str): Path to reenacted video
    
    Returns:
        dict: Quality assessment metrics
    """
    # Open videos
    cap_orig = cv2.VideoCapture(original_video_path)
    cap_reen = cv2.VideoCapture(reenacted_video_path)
    
    detector = FaceDetector.create("mediapipe")
    
    frame_similarities = []
    face_consistencies = []
    
    frame_count = 0
    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_reen, frame_reen = cap_reen.read()
        
        if not ret_orig or not ret_reen:
            break
        
        # Resize to same dimensions for comparison
        height, width = frame_orig.shape[:2]
        frame_reen = cv2.resize(frame_reen, (width, height))
        
        # Calculate structural similarity
        gray_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
        gray_reen = cv2.cvtColor(frame_reen, cv2.COLOR_BGR2GRAY)
        
        # Simple similarity metric (can be replaced with SSIM)
        similarity = cv2.matchTemplate(gray_orig, gray_reen, cv2.TM_CCOEFF_NORMED)[0][0]
        frame_similarities.append(similarity)
        
        # Check face consistency in reenacted video
        temp_frame_path = "temp_quality_frame.jpg"
        cv2.imwrite(temp_frame_path, frame_reen)
        
        detections = detector.detect(
            image_path=temp_frame_path,
            save_csv=False,
            save_annotated=False
        )
        
        face_consistencies.append(len(detections) == 1)  # Expect exactly one face
        frame_count += 1
        
        # Limit analysis to first 100 frames for speed
        if frame_count >= 100:
            break
    
    cap_orig.release()
    cap_reen.release()
    
    # Clean up
    import os
    if os.path.exists("temp_quality_frame.jpg"):
        os.remove("temp_quality_frame.jpg")
    
    # Calculate metrics
    avg_similarity = np.mean(frame_similarities) if frame_similarities else 0
    face_consistency_rate = np.mean(face_consistencies) if face_consistencies else 0
    
    # Overall quality score
    quality_score = (avg_similarity + face_consistency_rate) / 2
    
    return {
        'average_similarity': avg_similarity,
        'face_consistency_rate': face_consistency_rate,
        'quality_score': quality_score,
        'frames_analyzed': frame_count,
        'quality_rating': 'Excellent' if quality_score > 0.8 else 
                         'Good' if quality_score > 0.6 else 
                         'Fair' if quality_score > 0.4 else 'Poor'
    }

# Example usage
source_image = "assets/images/portrait.jpg"
driving_video = "assets/videos/expression_video.mp4"

# Generate reenactment
reenactor = FaceReenactor.create("tps")
result_path = reenactor.reenact_from_video(
    source_path=source_image,
    driving_video_path=driving_video,
    output_path="output/quality_test",
    save_comparison=True
)

# Assess quality
quality_metrics = assess_reenactment_quality(driving_video, result_path)

print("Reenactment Quality Assessment:")
for metric, value in quality_metrics.items():
    print(f"  {metric}: {value}")
```

## Error Handling and Troubleshooting

Robust reenactment with error handling:

```python
"""
Face reenactment with comprehensive error handling.
"""
from mukh.reenactment import FaceReenactor
import os
import cv2

def safe_reenactment(source_path, driving_video_path, output_path):
    """
    Perform face reenactment with comprehensive error handling.
    
    Args:
        source_path (str): Path to source image
        driving_video_path (str): Path to driving video
        output_path (str): Output directory path
    
    Returns:
        str or None: Path to result video or None if error
    """
    try:
        # Validate source image
        if not os.path.exists(source_path):
            print(f"Error: Source image '{source_path}' not found")
            return None
        
        # Check image format
        valid_image_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        source_ext = os.path.splitext(source_path)[1].lower()
        if source_ext not in valid_image_formats:
            print(f"Error: Unsupported image format '{source_ext}'")
            return None
        
        # Validate driving video
        if not os.path.exists(driving_video_path):
            print(f"Error: Driving video '{driving_video_path}' not found")
            return None
        
        # Check video format and properties
        cap = cv2.VideoCapture(driving_video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video '{driving_video_path}'")
            return None
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if frame_count < 10:
            print(f"Warning: Video is very short ({frame_count} frames)")
        
        if fps < 10 or fps > 60:
            print(f"Warning: Unusual FPS detected ({fps})")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize reenactor
        reenactor = FaceReenactor.create("tps")
        
        # Perform reenactment
        result_path = reenactor.reenact_from_video(
            source_path=source_path,
            driving_video_path=driving_video_path,
            output_path=output_path,
            save_comparison=True,
            resize_to_image_resolution=False
        )
        
        # Validate output
        if os.path.exists(result_path):
            file_size = os.path.getsize(result_path)
            if file_size < 1000:  # Less than 1KB indicates potential issue
                print(f"Warning: Output file is very small ({file_size} bytes)")
            else:
                print(f"✓ Reenactment successful: {result_path}")
                print(f"  Output file size: {file_size / (1024*1024):.2f} MB")
                return result_path
        else:
            print("Error: Output file was not created")
            return None
        
    except Exception as e:
        print(f"Error during reenactment: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Usage with error handling
source_images = [
    "assets/images/portrait1.jpg",
    "assets/images/portrait2.jpg",
    "non_existent_image.jpg"  # This will be handled gracefully
]

driving_videos = [
    "assets/videos/expression1.mp4",
    "assets/videos/expression2.mp4"
]

for i, source in enumerate(source_images):
    for j, driving in enumerate(driving_videos):
        output_dir = f"output/safe_reenactment/source_{i}_driving_{j}"
        
        print(f"\nProcessing source {i+1}, driving video {j+1}")
        result = safe_reenactment(source, driving, output_dir)
        
        if result:
            print(f"Success: {result}")
        else:
            print("Failed - skipping to next combination")
```

## Next Steps

- Explore [Deepfake Detection](deepfake-detection.md) to verify generated content
- Learn about [Face Detection](face-detection.md) for preprocessing
- Check the [API Reference](../api/face-reenactment.md) for detailed documentation
- Try combining reenactment with other Mukh features for advanced pipelines 