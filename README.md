# Mukh

<div align="center">

[![Downloads](https://static.pepy.tech/personalized-badge/mukh?period=total&units=international_system&left_color=grey&right_color=blue&left_text=downloads)](https://pepy.tech/project/mukh)
[![Documentation](https://img.shields.io/badge/docs-View%20Documentation-blue.svg?style=flat)](https://ishandutta0098.github.io/mukh/)
[![Stars](https://img.shields.io/github/stars/ishandutta0098/mukh?color=yellow&style=flat&label=%E2%AD%90%20stars)](https://github.com/ishandutta0098/mukh/stargazers)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg?style=flat)](https://github.com/ishandutta0098/mukh/blob/master/LICENSE)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-@ishandutta0098-blue.svg?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ishandutta0098)
[![Twitter](https://img.shields.io/:follow-@ishandutta0098-blue.svg?style=flat&logo=x)](https://twitter.com/intent/user?screen_name=ishandutta0098)
[![YouTube](https://img.shields.io/badge/YouTube-@ishandutta--ai-red?style=flat&logo=youtube)](https://www.youtube.com/@ishandutta-ai)

</div>

Mukh (मुख, meaning "face" in Sanskrit) is a comprehensive face analysis library that provides unified APIs for various face-related tasks. It simplifies the process of working with multiple face analysis models through a consistent interface.

## Features

- 🎯 **Unified API**: Single, consistent interface for multiple face analysis tasks
- 🔄 **Model Flexibility**: Support for multiple models per task
- 🛠️ **Custom Pipelines**: Optimized preprocessing and model combinations
- 📊 **Evaluator Mode**: Intelligent model recommendations based on your dataset
- 🚀 **Easy to Use**: Simple, intuitive APIs for quick integration

## Currently Supported Tasks

- Face Detection
- Facial Landmark Prediction

## Installation

```bash
pip install mukh
```

## Usage

### Face Detection

```python
from mukh.face_detection import FaceDetector

# Initialize detector
detection_model = "mediapipe" # Available models: "blazeface", "mediapipe", "ultralight"
detector = FaceDetector.create(detection_model)

# Detect faces
detections = detector.detect(
    image_path="assets/images/img1.jpg",                       # Path to the image to detect faces in
    save_csv=True,                                             # Save the detections to a CSV file
    csv_path=f"output/{detection_model}/detections.csv",       # Path to save the CSV file
    save_annotated=True,                                       # Save the annotated image
    output_folder=f"output/detection_model",                   # Path to save the annotated image
)
```
  
### Face Reenactment

```python
from mukh.face_reenactment import FaceReenactor

# Initialize reenactor
reenactor_model = "tps" # Available models: "tps"
reenactor = FaceReenactor.create(reenactor_model)

# Reenact face
result_path = reenactor.reenact_from_video(
    source_path="assets/images/img1.jpg",         # Path to the source image
    driving_video_path="assets/videos/video.mp4", # Path to the driving video
    output_path=f"output/{reenactor_model}",      # Path to save the reenacted video
    save_comparison=True,                         # Save the comparison video
    resize_to_image_resolution=False,             # Resize the reenacted video to the image resolution
)
```

## Contact

For questions and feedback, please open an issue on GitHub.
