# Quick Start Guide

## Face Detection

```python
from mukh.face_detection import FaceDetector

detector = FaceDetector.create("mediapipe")
detections = detector.detect("path/to/image.jpg")
```

## Face Reenactment

```python
from mukh.face_reenactment import FaceReenactor

reenactor = FaceReenactor.create("tps")
result = reenactor.reenact_from_video(
    source_path="source.jpg",
    driving_video_path="driving.mp4",
    output_path="output/"
)
```

## Deepfake Detection

```python
from mukh.deepfake_detection import DeepfakeDetector

detector = DeepfakeDetector("efficientnet")
detections, result = detector.detect("media.jpg")
```

## Pipeline

```python
from mukh.pipelines.deepfake_detection import PipelineDeepfakeDetection

model_configs = {"resnet_inception": 0.5, "efficientnet": 0.5}
pipeline = PipelineDeepfakeDetection(model_configs)
result = pipeline.detect("media.jpg")
```

## Next Steps

- [Examples](../examples/face-detection.md)
- [API Reference](../api/core.md) 