"""
BlazeFace PyTorch implementation adapted from:
https://github.com/hollance/BlazeFace-PyTorch

Original implementation by M.I. Hollemans
"""

from typing import List, Tuple

import cv2
import numpy as np
import torch

from ....core.types import BoundingBox, FaceDetection
from ..base_detector import BaseFaceDetector
from .blazeface_torch import BlazeFace


class BlazeFaceDetector(BaseFaceDetector):
    def __init__(
        self,
        weights_path: str = "mukh/detection/models/blazeface/blazeface.pth",
        anchors_path: str = "mukh/detection/models/blazeface/anchors.npy",
        confidence_threshold: float = 0.75,
    ):
        super().__init__(confidence_threshold)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = BlazeFace().to(self.device)
        self.net.load_weights(weights_path)
        self.net.load_anchors(anchors_path)

    def detect(self, image_path: str) -> List[FaceDetection]:
        # Load image from path
        image = self._load_image(image_path)

        # Get original dimensions
        orig_h, orig_w = image.shape[:2]

        # Resize to 128x128
        image_resized = cv2.resize(image, (128, 128))

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        # Get detections
        detections = self.net.predict_on_image(image_rgb)

        # Convert to FaceDetection objects
        faces = []
        for detection in detections:
            # Convert normalized coordinates back to original image size
            x1 = float(detection[1]) * orig_w  # xmin
            y1 = float(detection[0]) * orig_h  # ymin
            x2 = float(detection[3]) * orig_w  # xmax
            y2 = float(detection[2]) * orig_h  # ymax

            bbox = BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2, confidence=float(detection[16])
            )

            # Extract landmarks and scale back to original size
            landmarks = []
            for i in range(6):
                x = float(detection[4 + i * 2]) * orig_w
                y = float(detection[4 + i * 2 + 1]) * orig_h
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
