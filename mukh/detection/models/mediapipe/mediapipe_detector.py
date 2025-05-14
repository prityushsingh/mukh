from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from ....core.types import BoundingBox, FaceDetection
from ..base_detector import BaseFaceDetector


class MediaPipeFaceDetector(BaseFaceDetector):
    def __init__(self, confidence_threshold: float = 0.5, model_selection: int = 0):
        super().__init__(confidence_threshold)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=confidence_threshold,
            model_selection=model_selection,
        )

    def detect(self, image_path: str) -> List[FaceDetection]:
        # Load image from path
        image = self._load_image(image_path)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image
        results = self.face_detection.process(image_rgb)

        faces = []
        if results.detections:
            image_height, image_width, _ = image.shape
            for detection in results.detections:
                bbox_rel = detection.location_data.relative_bounding_box

                # Convert relative coordinates to absolute pixels
                x = int(bbox_rel.xmin * image_width)
                y = int(bbox_rel.ymin * image_height)
                w = int(bbox_rel.width * image_width)
                h = int(bbox_rel.height * image_height)

                bbox = BoundingBox(
                    x1=x, y1=y, x2=x + w, y2=y + h, confidence=detection.score[0]
                )

                # Extract landmarks
                landmarks = []
                for landmark in detection.location_data.relative_keypoints:
                    x = int(landmark.x * image_width)
                    y = int(landmark.y * image_height)
                    landmarks.append([x, y])

                faces.append(FaceDetection(bbox=bbox, landmarks=np.array(landmarks)))

        return faces

    def detect_with_landmarks(
        self, image_path: str
    ) -> Tuple[List[FaceDetection], np.ndarray]:
        # Load image and detect faces
        image = self._load_image(image_path)
        faces = self.detect(image_path)

        # Draw detections on image copy
        annotated_image = self._draw_detections(image, faces)
        return faces, annotated_image

    def _draw_detections(
        self, image: np.ndarray, faces: List[FaceDetection]
    ) -> np.ndarray:
        image_copy = image.copy()
        for face in faces:
            bbox = face.bbox
            # Draw bounding box
            cv2.rectangle(
                image_copy,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                (0, 255, 0),
                2,
            )

            # Draw landmarks
            if face.landmarks is not None:
                for x, y in face.landmarks:
                    cv2.circle(image_copy, (int(x), int(y)), 2, (0, 255, 0), 2)

        return image_copy
