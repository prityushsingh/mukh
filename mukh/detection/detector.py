from typing import List, Literal

from .models.base_detector import BaseFaceDetector
from .models.blazeface import BlazeFaceDetector
from .models.mediapipe import MediaPipeFaceDetector
from .models.ultralight import UltralightDetector

DetectorType = Literal["blazeface", "mediapipe", "ultralight"]


class FaceDetector:
    """Unified interface for face detection models."""

    @staticmethod
    def create(model: DetectorType, **kwargs) -> BaseFaceDetector:
        """Create a face detector instance."""
        detectors = {
            "blazeface": BlazeFaceDetector,
            "mediapipe": MediaPipeFaceDetector,
            "ultralight": UltralightDetector,
        }

        if model not in detectors:
            raise ValueError(
                f"Unknown detector model: {model}. "
                f"Available models: {list(detectors.keys())}"
            )

        return detectors[model](**kwargs)

    @staticmethod
    def list_available_models() -> List[str]:
        """Return list of available detection models."""
        return ["blazeface", "mediapipe", "ultralight"]
