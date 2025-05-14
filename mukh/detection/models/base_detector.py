from abc import ABC, abstractmethod
from typing import List, Tuple

import cv2
import numpy as np

from ...core.types import FaceDetection


class BaseFaceDetector(ABC):
    """Abstract base class for all face detector implementations."""

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from path and convert to BGR format."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from path: {image_path}")
        return image

    @abstractmethod
    def detect(self, image_path: str) -> List[FaceDetection]:
        """
        Detect faces in the image at the given path.

        Args:
            image_path: Path to the input image

        Returns:
            List of FaceDetection objects
        """
        pass

    @abstractmethod
    def detect_with_landmarks(
        self, image_path: str
    ) -> Tuple[List[FaceDetection], np.ndarray]:
        """
        Detect faces and return annotated image with landmarks.

        Args:
            image_path: Path to the input image

        Returns:
            Tuple containing:
            - List of FaceDetection objects
            - Annotated image with detections drawn
        """
        pass
