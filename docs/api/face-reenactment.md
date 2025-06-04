# Face Reenactment API

The face reenactment module provides functionality for animating faces using driving videos.

## FaceReenactor

::: mukh.reenactment.FaceReenactor
    options:
      show_source: false
      heading_level: 3

## Examples

### Basic Face Reenactment

```python
from mukh.reenactment import FaceReenactor

# Create reenactor with TPS model
reenactor = FaceReenactor.create("tps")

# Reenact face from video
result_path = reenactor.reenact_from_video(
    source_path="source_image.jpg",
    driving_video_path="driving_motion.mp4",
    output_path="output/reenacted_video.mp4",
    save_comparison=True,  # Save side-by-side comparison
    resize_to_image_resolution=False
)

print(f"Reenacted video saved to: {result_path}")
```

### Batch Reenactment

```python
import os
from mukh.reenactment import FaceReenactor

reenactor = FaceReenactor.create("tps")

# Multiple source images with same driving video
source_images = ["person1.jpg", "person2.jpg", "person3.jpg"]
driving_video = "expression_sequence.mp4"

for source_image in source_images:
    name = os.path.splitext(source_image)[0]
    result_path = reenactor.reenact_from_video(
        source_path=source_image,
        driving_video_path=driving_video,
        output_path=f"output/reenacted_{name}.mp4",
        save_comparison=True
    )
    print(f"Processed {source_image} -> {result_path}")
```

### Custom Resolution and Quality Settings

```python
from mukh.reenactment import FaceReenactor

reenactor = FaceReenactor.create("tps")

# High quality reenactment with custom settings
result_path = reenactor.reenact_from_video(
    source_path="high_res_source.jpg",
    driving_video_path="driving_sequence.mp4",
    output_path="output/high_quality_result.mp4",
    save_comparison=True,
    resize_to_image_resolution=True,  # Maintain source image resolution
)

print(f"High quality result: {result_path}")
```

### Video-to-Video Reenactment

```python
from mukh.reenactment import FaceReenactor

reenactor = FaceReenactor.create("tps")

# Use first frame of a video as source
result_path = reenactor.reenact_from_video(
    source_path="source_video.mp4",  # Will use first frame
    driving_video_path="expressions.mp4",
    output_path="output/video_to_video_result.mp4",
    save_comparison=True
)

print(f"Video-to-video reenactment saved: {result_path}")
```

## Model Parameters

### TPS Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_kp` | int | 10 | Number of keypoints for motion representation |
| `num_channels` | int | 3 | Number of image channels (RGB) |
| `estimate_jacobian` | bool | True | Estimate Jacobian for better transformations |
| `temperature` | float | 0.1 | Temperature for keypoint detection |
| `block_expansion` | int | 32 | Feature block expansion factor |
| `max_features` | int | 1024 | Maximum number of features to extract |
| `scale_factor` | float | 0.25 | Scale factor for feature pyramid |
| `num_down_blocks` | int | 2 | Number of downsampling blocks |
| `num_bottleneck_blocks` | int | 6 | Number of bottleneck blocks |
| `estimate_occlusion_map` | bool | True | Estimate occlusion for better handling |

## Output Options

### Video Quality Settings

```python
# High quality output
result_path = reenactor.reenact_from_video(
    source_path="source.jpg",
    driving_video_path="driving.mp4",
    output_path="output/hq",
    fps=30,                    # High frame rate
    quality='high',           # High quality encoding
    codec='h264',             # Efficient codec
    bitrate='5M',             # High bitrate
    resize_to_image_resolution=True  # Match source resolution
)

# Fast/preview quality
result_path = reenactor.reenact_from_video(
    source_path="source.jpg",
    driving_video_path="driving.mp4",
    output_path="output/preview",
    fps=15,                   # Lower frame rate
    quality='medium',         # Medium quality
    codec='mpeg4',            # Fast codec
    max_resolution=(256, 256) # Lower resolution for speed
)
```

### Comparison Videos

```python
# Generate side-by-side comparison
result_path = reenactor.reenact_from_video(
    source_path="source.jpg",
    driving_video_path="driving.mp4",
    output_path="output/comparison",
    save_comparison=True,         # Save comparison video
    comparison_layout='horizontal', # 'horizontal' or 'vertical'
    add_labels=True,             # Add text labels
    label_font_size=20           # Font size for labels
)
```

## Error Handling

```python
from mukh.reenactment import FaceReenactor
import os

def safe_reenactment(source_path, driving_video_path, output_path):
    """Perform reenactment with error handling."""
    try:
        # Validate inputs
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source image not found: {source_path}")
        
        if not os.path.exists(driving_video_path):
            raise FileNotFoundError(f"Driving video not found: {driving_video_path}")
        
        # Check file formats
        valid_image_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        valid_video_formats = ['.mp4', '.avi', '.mov', '.mkv']
        
        source_ext = os.path.splitext(source_path)[1].lower()
        video_ext = os.path.splitext(driving_video_path)[1].lower()
        
        if source_ext not in valid_image_formats:
            raise ValueError(f"Unsupported image format: {source_ext}")
        
        if video_ext not in valid_video_formats:
            raise ValueError(f"Unsupported video format: {video_ext}")
        
        # Initialize reenactor
        reenactor = FaceReenactor.create("tps")
        
        # Perform reenactment
        result_path = reenactor.reenact_from_video(
            source_path=source_path,
            driving_video_path=driving_video_path,
            output_path=output_path,
            save_comparison=True
        )
        
        return result_path
        
    except Exception as e:
        print(f"Error during reenactment: {e}")
        return None

# Usage
result = safe_reenactment("face.jpg", "expressions.mp4", "output")
if result:
    print(f"Success: {result}")
else:
    print("Reenactment failed")
```

## Performance Tips

### Optimization for Speed

```python
# Faster processing settings
reenactor = FaceReenactor.create(
    model_name="tps",
    num_kp=10,                # Fewer keypoints
    max_features=512,         # Fewer features
    estimate_jacobian=False,  # Skip Jacobian estimation
    estimate_occlusion_map=False  # Skip occlusion estimation
)

# Process at lower resolution
result_path = reenactor.reenact_from_video(
    source_path="source.jpg",
    driving_video_path="driving.mp4",
    output_path="output/fast",
    max_resolution=(256, 256),  # Lower resolution
    fps=15,                     # Lower frame rate
    quality='medium'            # Medium quality
)
```

### Memory Management

```python
import gc
from mukh.reenactment import FaceReenactor

# Process multiple videos with memory cleanup
source_image = "portrait.jpg"
driving_videos = ["video1.mp4", "video2.mp4", "video3.mp4"]

for i, video_path in enumerate(driving_videos):
    # Create fresh reenactor instance
    reenactor = FaceReenactor.create("tps")
    
    # Process video
    result_path = reenactor.reenact_from_video(
        source_path=source_image,
        driving_video_path=video_path,
        output_path=f"output/batch_{i}"
    )
    
    # Clean up memory
    del reenactor
    gc.collect()
    
    print(f"Processed {video_path} -> {result_path}")
``` 