# Mukh - Face Analysis Library

<div align="center" markdown>

[![Downloads](https://static.pepy.tech/personalized-badge/mukh?period=total&units=international_system&left_color=grey&right_color=blue&left_text=downloads)](https://pepy.tech/project/mukh)
[![Stars](https://img.shields.io/github/stars/ishandutta0098/mukh?color=yellow&style=flat&label=%E2%AD%90%20stars)](https://github.com/ishandutta0098/mukh/stargazers)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg?style=flat)](https://github.com/ishandutta0098/mukh/blob/master/LICENSE)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-@ishandutta0098-blue.svg?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ishandutta0098)
[![Twitter](https://img.shields.io/:follow-@ishandutta0098-blue.svg?style=flat&logo=x)](https://twitter.com/intent/user?screen_name=ishandutta0098)
[![YouTube](https://img.shields.io/badge/YouTube-@ishandutta--ai-red?style=flat&logo=youtube)](https://www.youtube.com/@ishandutta-ai)

</div>

Mukh (मुख, meaning "face" in Sanskrit) is a comprehensive face analysis library with unified APIs for face detection, reenactment, and deepfake detection.

## Features

- Face Detection
- Face Reenactment  
- Deepfake Detection
- Pipelines
    - Deepfake Detection Pipeline

## Quick Start

```python
from mukh.face_detection import FaceDetector

detector = FaceDetector.create("mediapipe")
detections = detector.detect("path/to/image.jpg")
```

## Documentation

- [Installation](getting-started/installation.md)
- [Examples](examples/face-detection.md)
- [API Reference](api/core.md)

## License

Apache 2.0 