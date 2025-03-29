import mediapipe as mp
import cv2
import numpy as np
from typing import List, Tuple, Union

class MediaPipeFaceDetector:
    """Face detector using MediaPipe Face Detection model."""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 model_selection: int = 0):
        """
        Initialize MediaPipe Face Detector.
        
        Args:
            min_detection_confidence: Minimum confidence value for face detection
            model_selection: 0 for short-range detection (2 meters), 1 for full-range detection (5 meters)
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection
        )
    
    def detect_faces(self, 
                    image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image in BGR format (OpenCV default)
            
        Returns:
            List of tuples containing:
            - Face bounding box (x, y, width, height)
            - Detection confidence
        """
        # Convert BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect faces
        results = self.face_detection.process(image_rgb)
        
        faces = []
        if results.detections:
            image_height, image_width, _ = image.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute pixels
                x = int(bbox.xmin * image_width)
                y = int(bbox.ymin * image_height)
                w = int(bbox.width * image_width)
                h = int(bbox.height * image_height)
                
                # Add face detection results to list
                faces.append(((x, y, w, h), detection.score[0]))
        
        return faces
    
    def draw_detections(self, 
                       image: np.ndarray,
                       faces: List[Tuple[Tuple[int, int, int, int], float]]) -> np.ndarray:
        """
        Draw face detections on an image.
        
        Args:
            image: Input image in BGR format
            faces: List of face detections from detect_faces()
            
        Returns:
            Image with drawn face detections
        """
        image_copy = image.copy()
        
        for (x, y, w, h), confidence in faces:
            # Draw bounding box
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"Face: {confidence:.2f}"
            cv2.putText(image_copy, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image_copy

    def __call__(self, 
                 image: np.ndarray,
                 return_annotated: bool = False) -> Union[
                     List[Tuple[Tuple[int, int, int, int], float]],
                     Tuple[List[Tuple[Tuple[int, int, int, int], float]], np.ndarray]
                 ]:
        """
        Detect faces in an image and optionally draw the results.
        
        Args:
            image: Input image in BGR format
            return_annotated: If True, returns both detections and annotated image
            
        Returns:
            If return_annotated is False: List of face detections
            If return_annotated is True: Tuple of (face detections, annotated image)
        """
        faces = self.detect_faces(image)
        
        if return_annotated:
            annotated_image = self.draw_detections(image, faces)
            return faces, annotated_image
        
        return faces
