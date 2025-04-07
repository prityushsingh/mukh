from typing import Literal, List
from .models.blazeface import BlazeFaceDetector
from .models.mediapipe import MediaPipeFaceDetector
from .models.base_detector import BaseFaceDetector

DetectorType = Literal["blazeface", "mediapipe"]

class FaceDetector:
    """Unified interface for face detection models."""
    
    @staticmethod
    def create(model: DetectorType, **kwargs) -> BaseFaceDetector:
        """Create a face detector instance."""
        detectors = {
            "blazeface": BlazeFaceDetector,
            "mediapipe": MediaPipeFaceDetector
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
        return ["blazeface", "mediapipe"] 