from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Union
from ...core.types import FaceDetection

class BaseFaceDetector(ABC):
    """Abstract base class for all face detector implementations."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in the given image.
        
        Args:
            image: Input image in BGR format (OpenCV default)
            
        Returns:
            List of FaceDetection objects
        """
        pass
    
    @abstractmethod
    def detect_with_landmarks(self, 
                            image: np.ndarray) -> Tuple[List[FaceDetection], np.ndarray]:
        """
        Detect faces and return annotated image with landmarks.
        
        Args:
            image: Input image in BGR format (OpenCV default)
            
        Returns:
            Tuple containing:
            - List of FaceDetection objects
            - Annotated image with detections drawn
        """
        pass 