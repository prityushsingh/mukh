# Face Reenactment Examples

## Basic Reenactment

```python
from mukh.reenactment import FaceReenactor

reenactor = FaceReenactor.create("tps")  # Available models: "tps"

result_path = reenactor.reenact_from_video(
    source_path="assets/images/img1.jpg",
    driving_video_path="assets/videos/video_1sec.mp4",
    output_path="output/tps",
    save_comparison=True,
    resize_to_image_resolution=False
)

print(f"Reenacted video saved to: {result_path}")
```

## Available Models

- `tps`

## Parameters

- `source_path`: Path to source image
- `driving_video_path`: Path to driving video
- `output_path`: Output directory
- `save_comparison`: Save side-by-side comparison video
- `resize_to_image_resolution`: Resize output to match source image resolution 