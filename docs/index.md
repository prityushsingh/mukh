# Mukh - Face Analysis Library

<div align="center" markdown>

[![Downloads](https://static.pepy.tech/personalized-badge/mukh?period=total&units=international_system&left_color=grey&right_color=blue&left_text=downloads)](https://pepy.tech/project/mukh)
[![Documentation](https://img.shields.io/badge/docs-View%20Documentation-blue.svg?style=flat)](https://ishandutta0098.github.io/mukh/)
[![Stars](https://img.shields.io/github/stars/ishandutta0098/mukh?color=yellow&style=flat&label=%E2%AD%90%20stars)](https://github.com/ishandutta0098/mukh/stargazers)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg?style=flat)](https://github.com/ishandutta0098/mukh/blob/master/LICENSE)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-@ishandutta0098-blue.svg?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ishandutta0098)
[![Twitter](https://img.shields.io/:follow-@ishandutta0098-blue.svg?style=flat&logo=x)](https://twitter.com/intent/user?screen_name=ishandutta0098)
[![YouTube](https://img.shields.io/badge/YouTube-@ishandutta--ai-red?style=flat&logo=youtube)](https://www.youtube.com/@ishandutta-ai)

</div>

Mukh (मुख, meaning "face" in Sanskrit) is a comprehensive face analysis library that provides unified APIs for various face-related tasks. It simplifies the process of working with multiple face analysis models through a consistent interface.

## :sparkles: Features

- :dart: **Unified API**: Single, consistent interface for multiple face analysis tasks
- :arrows_counterclockwise: **Model Flexibility**: Support for multiple models per task
- :hammer_and_wrench: **Custom Pipelines**: Optimized preprocessing and model combinations
- :bar_chart: **Evaluator Mode**: Intelligent model recommendations based on your dataset
- :rocket: **Easy to Use**: Simple, intuitive APIs for quick integration

## :white_check_mark: Currently Supported Tasks

- **Face Detection**: Detect faces in images using multiple state-of-the-art models
- **Facial Landmark Prediction**: Predict facial landmarks with high accuracy
- **Face Reenactment**: Generate realistic face reenactment videos
- **Deepfake Detection**: Detect deepfake content in images and videos

## :zap: Quick Start

Get started with Mukh in just a few lines of code:

```python
from mukh.face_detection import FaceDetector

# Initialize detector
detector = FaceDetector.create("mediapipe")

# Detect faces
detections = detector.detect(
    image_path="path/to/your/image.jpg",
    save_csv=True,
    save_annotated=True
)
```

## :books: Documentation Structure

- **[Getting Started](getting-started/installation.md)**: Installation and setup instructions
- **[Examples](examples/face-detection.md)**: Practical examples for each feature
- **[API Reference](api/core.md)**: Complete API documentation
- **[Contributing](contributing.md)**: How to contribute to the project

## :handshake: Community & Support

- :bug: **Found a bug?** [Open an issue](https://github.com/ishandutta0098/mukh/issues)
- :bulb: **Have an idea?** [Start a discussion](https://github.com/ishandutta0098/mukh/discussions)
- :speech_balloon: **Need help?** Check our [documentation](getting-started/installation.md) or ask questions

## :memo: License

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://github.com/ishandutta0098/mukh/blob/master/LICENSE) file for details. 