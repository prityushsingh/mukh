"""
mukh - A python package to perform a variety of tasks on face images
"""

__version__ = "0.0.3"

# from typing import Literal
# from .base import BaseDetector
# from .models.blazeface import BlazeFaceDetector

# DetectorType = Literal["blazeface"]

# class FaceDetectorFactory:
#     """Factory class for creating face detectors."""
    
#     @staticmethod
#     def create(model: DetectorType, **kwargs) -> BaseDetector:
#         """
#         Create a face detector instance.
        
#         Args:
#             model: Name of the detection model
#             **kwargs: Model-specific parameters
            
#         Returns:
#             BaseDetector instance
#         """
#         detectors = {
#             "blazeface": BlazeFaceDetector
#         }
        
#         if model not in detectors:
#             raise ValueError(
#                 f"Unknown detector model: {model}. "
#                 f"Available models: {list(detectors.keys())}"
#             )
            
#         return detectors[model](**kwargs)