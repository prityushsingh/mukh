"""EfficientNet deepfake detection model implementation.

This module implements a deepfake detection model using EfficientNet architecture
based on the fornet library implementation, with explainability features through GradCAM visualization.

GitHub: https://github.com/polimi-ispl/icpr2020dfdc
"""

import os
from typing import List, Optional

import cv2
import face_recognition
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from ....core.types import DeepfakeDetection
from ..base import BaseDeepfakeDetector


class EfficientNetDetector(BaseDeepfakeDetector):
    """EfficientNet deepfake detector implementation.

    A deepfake detection model using EfficientNet architecture based on the fornet
    library implementation with explainability features through GradCAM visualization.

    Attributes:
        device: PyTorch device (CPU/CUDA) for model execution
        model: The EfficientNet model for deepfake detection
        confidence_threshold: Minimum confidence for valid detections
        face_size: Input face size for preprocessing
        transform: Image preprocessing transforms
        face_policy: Face extraction policy ('scale' or 'tight')
        mean: ImageNet normalization mean values
        std: ImageNet normalization std values
    """

    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = 0.5,
        device: str = None,
        net_model: str = "EfficientNetAutoAttB4",
        train_db: str = "DFDC",
        face_policy: str = "scale",
        face_size: int = 224,
    ):
        """Initializes the EfficientNet deepfake detector.

        Args:
            model_path: Path to the trained model checkpoint (if None, downloads from fornet)
            confidence_threshold: Minimum confidence threshold for detections
            device: Device to run inference on ('cpu' or 'cuda'). Auto-detected if None
            net_model: EfficientNet model variant ('EfficientNetB4', 'EfficientNetAutoAttB4', etc.)
            train_db: Training database ('DFDC', 'FFPP')
            face_policy: Face extraction policy ('scale' or 'tight')
            face_size: Input face size for preprocessing
        """
        super().__init__(confidence_threshold)

        # Set device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model configuration
        self.net_model = net_model
        self.train_db = train_db
        self.face_policy = face_policy
        self.face_size = face_size

        # ImageNet normalization parameters
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Initialize the model
        self.model = self._create_model()

        # Load the trained weights
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
        else:
            # Load from fornet weights if available
            self._load_fornet_weights()

        self.model.to(self.device)
        self.model.eval()

        # Initialize transforms
        self.transform = self._get_transformer()

    def _create_model(self) -> nn.Module:
        """Creates the EfficientNet model architecture.

        Returns:
            EfficientNet model configured for binary classification
        """
        try:
            # Try to import fornet architectures
            from mukh.deepfake_detection.models.efficientnet.architectures import fornet

            model = getattr(fornet, self.net_model)()
            return model
        except ImportError:
            # Fallback to standard EfficientNet if fornet is not available
            from efficientnet_pytorch import EfficientNet

            print("Warning: fornet not available, using standard EfficientNet")
            model = EfficientNet.from_pretrained("efficientnet-b4", num_classes=1)
            return model

    def _load_fornet_weights(self) -> None:
        """Loads pre-trained weights from fornet library."""
        try:
            from torch.utils.model_zoo import load_url

            from mukh.deepfake_detection.models.efficientnet.architectures import (
                weights,
            )

            model_url = weights.weight_url[f"{self.net_model}_{self.train_db}"]
            state_dict = load_url(model_url, map_location=self.device, check_hash=True)
            self.model.load_state_dict(state_dict)
            print(f"Loaded fornet weights for {self.net_model}_{self.train_db}")
        except Exception as e:
            print(f"Warning: Could not load fornet weights: {e}")

    def _get_transformer(self) -> transforms.Compose:
        """Gets the image transformer based on face policy and model normalizer.

        Returns:
            Composed transforms for image preprocessing
        """
        try:
            # Try to use fornet utils if available
            from isplutils import utils

            normalizer = (
                self.model.get_normalizer()
                if hasattr(self.model, "get_normalizer")
                else None
            )
            return utils.get_transformer(
                self.face_policy, self.face_size, normalizer, train=False
            )
        except ImportError:
            # Fallback to standard transforms
            return transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((self.face_size, self.face_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )

    def _extract_face_blazeface(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face using BlazeFace detector (if available).

        Args:
            image: Input image as numpy array.

        Returns:
            numpy.ndarray: Cropped face image, or None if no face found.
        """
        try:
            from mukh.core import download_blazeface_models
            from mukh.face_detection.models.blazeface import BlazeFace, FaceExtractor

            # Initialize BlazeFace if not already done
            if not hasattr(self, "face_extractor"):
                facedet = BlazeFace().to(self.device)
                # Download and get paths for the BlazeFace models
                weights_path, anchors_path = download_blazeface_models()
                facedet.load_weights(weights_path)
                facedet.load_anchors(anchors_path)
                self.face_extractor = FaceExtractor(facedet=facedet)

            # Convert numpy to PIL if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Extract faces
            faces_data = self.face_extractor.process_image(img=image)

            if faces_data["faces"]:
                # Return the face with highest confidence
                return faces_data["faces"][0]
            else:
                return None

        except ImportError:
            print("BlazeFace not available, skipping BlazeFace extraction")
            return None
        except Exception as e:
            print(f"BlazeFace extraction failed: {e}")
            return None

    def _extract_face_recognition(self, image: np.ndarray) -> np.ndarray:
        """Extract face using face_recognition library (fallback).

        Args:
            image: Input image as numpy array.

        Returns:
            numpy.ndarray: Cropped face image, or original image if no face found.
        """
        try:
            # Convert PIL to numpy if needed
            if isinstance(image, Image.Image):
                image = np.array(image)

            # Find face locations
            faces = face_recognition.face_locations(image)

            if faces:
                # Use the first detected face
                top, right, bottom, left = faces[0]
                face = image[top:bottom, left:right, :]
                return face
            else:
                print("Warning: No face detected, using full image")
                return image

        except Exception as e:
            print(f"Error in face extraction: {e}")
            return image

    def _extract_face(self, image: np.ndarray) -> np.ndarray:
        """Extract face from image using available face detectors.

        Args:
            image: Input image as numpy array.

        Returns:
            numpy.ndarray: Cropped face image.
        """
        # Try BlazeFace first (more accurate for deepfake detection)
        face = self._extract_face_blazeface(image)

        if face is not None:
            return np.array(face)
        else:
            # Fallback to face_recognition
            return self._extract_face_recognition(image)

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocesses image for model input.

        Args:
            image: Input image as numpy array

        Returns:
            Preprocessed tensor ready for model input
        """
        # Extract face
        face_image = self._extract_face(image)

        # Convert to PIL Image if needed
        if isinstance(face_image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_image = Image.fromarray(face_image)

        # Apply transforms
        if hasattr(self.transform, "__call__"):
            # For fornet transforms that expect dict format
            try:
                transformed = self.transform(image=np.array(face_image))
                if isinstance(transformed, dict) and "image" in transformed:
                    tensor = transformed["image"]
                else:
                    tensor = transformed
            except:
                # Fallback to standard transform
                tensor = transforms.Compose(
                    [
                        transforms.Resize((self.face_size, self.face_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(self.mean, self.std),
                    ]
                )(face_image)
        else:
            tensor = self.transform(face_image)

        # Add batch dimension if needed
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def detect_image(
        self,
        image_path: str,
        save_csv: bool = False,
        csv_path: str = "deepfake_detections.csv",
        save_annotated: bool = False,
        output_folder: str = "output",
    ) -> DeepfakeDetection:
        """Detects deepfake in the given image.

        Args:
            image_path: Path to the input image
            save_csv: Whether to save detection results to CSV file
            csv_path: Path where to save the CSV file
            save_annotated: Whether to save annotated image with results
            output_folder: Folder path where to save annotated images

        Returns:
            DeepfakeDetection object containing detection results
        """
        # Load image
        image = self._load_image(image_path)

        # Preprocess image
        input_tensor = self._preprocess_image(image)

        # Make prediction
        with torch.no_grad():
            output = self.model(input_tensor)

            # Handle different output formats
            if isinstance(output, tuple):
                logits = output[1] if len(output) > 1 else output[0]
            else:
                logits = output

            # Apply sigmoid for binary classification
            if logits.shape[-1] == 1:
                prob = torch.sigmoid(logits).item()
                is_deepfake = prob >= self.confidence_threshold
                confidence = prob if is_deepfake else (1 - prob)
            else:
                probs = torch.softmax(logits, dim=1)
                fake_prob = probs[0, 1].item()
                is_deepfake = fake_prob >= self.confidence_threshold
                confidence = fake_prob if is_deepfake else (1 - fake_prob)

            detection = DeepfakeDetection(
                frame_number=0,  # Single image, so frame 0
                is_deepfake=is_deepfake,
                confidence=confidence,
                model_name=f"EfficientNet-{self.net_model}",
            )

        # Save explainability visualization if requested
        if save_annotated:
            self._save_explainability_visualization(
                input_tensor, image, image_path, output_folder
            )
            self._save_annotated_image(image, detection, image_path, output_folder)

        # Save to CSV if requested
        if save_csv:
            self._save_detections_to_csv(detection, image_path, csv_path)

        return detection

    def detect_video(
        self,
        video_path: str,
        save_csv: bool = False,
        csv_path: str = "deepfake_detections.csv",
        save_annotated: bool = False,
        output_folder: str = "output",
        num_frames: int = 11,
    ) -> List[DeepfakeDetection]:
        """Detects deepfake in the given video using equally spaced frames.

        Args:
            video_path: Path to the input video
            save_csv: Whether to save detection results to CSV file
            csv_path: Path where to save the CSV file
            save_annotated: Whether to save annotated video with results
            output_folder: Folder path where to save annotated videos
            num_frames: Number of equally spaced frames to analyze (default: 11)

        Returns:
            List of DeepfakeDetection objects for analyzed frames
        """
        # Extract equally spaced frames
        extracted_frames = self._extract_equally_spaced_frames(video_path, num_frames)

        detections = []

        for frame_number, frame in extracted_frames:
            try:
                # Preprocess frame
                input_tensor = self._preprocess_image(frame)

                # Make prediction
                with torch.no_grad():
                    output = self.model(input_tensor)

                    # Handle different output formats
                    if isinstance(output, tuple):
                        logits = output[1] if len(output) > 1 else output[0]
                    else:
                        logits = output

                    # Apply sigmoid for binary classification
                    if logits.shape[-1] == 1:
                        prob = torch.sigmoid(logits).item()
                        is_deepfake = prob >= self.confidence_threshold
                        confidence = prob if is_deepfake else (1 - prob)
                    else:
                        probs = torch.softmax(logits, dim=1)
                        fake_prob = probs[0, 1].item()
                        is_deepfake = fake_prob >= self.confidence_threshold
                        confidence = fake_prob if is_deepfake else (1 - fake_prob)

                    detection = DeepfakeDetection(
                        frame_number=frame_number,
                        is_deepfake=is_deepfake,
                        confidence=confidence,
                        model_name=f"EfficientNet-{self.net_model}",
                    )

                    detections.append(detection)

            except Exception as e:
                print(f"Error processing frame {frame_number}: {e}")
                # Skip frames with errors

        # Save annotated video if requested
        if save_annotated and detections:
            self._save_annotated_video(video_path, detections, output_folder)

        # Save to CSV if requested
        if save_csv and detections:
            self._save_detections_to_csv(detections, video_path, csv_path)

        return detections

    def _save_explainability_visualization(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        image_path: str,
        output_folder: str,
    ) -> None:
        """Saves explainability visualization using GradCAM.

        Args:
            input_tensor: Preprocessed input tensor for model
            original_image: Original image for visualization overlay
            image_path: Path to the original image
            output_folder: Directory to save visualizations
        """
        try:
            # Create output directory
            os.makedirs(output_folder, exist_ok=True)

            # Find the last convolutional layer for GradCAM
            target_layers = []
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.AdaptiveAvgPool2d)):
                    target_layers = [module]

            if not target_layers:
                print("Warning: No suitable layer found for GradCAM")
                return

            # Generate GradCAM visualization
            cam = GradCAM(model=self.model, target_layers=target_layers)
            targets = [ClassifierOutputTarget(0)]

            grayscale_cam = cam(
                input_tensor=input_tensor, targets=targets, eigen_smooth=True
            )
            grayscale_cam = grayscale_cam[0, :]

            # Prepare image for visualization
            face_image = self._extract_face(original_image)
            image_for_viz = cv2.resize(face_image, (self.face_size, self.face_size))

            # Convert BGR to RGB if needed
            if len(image_for_viz.shape) == 3 and image_for_viz.shape[2] == 3:
                image_for_viz = cv2.cvtColor(image_for_viz, cv2.COLOR_BGR2RGB)

            image_for_viz = image_for_viz.astype(np.float32) / 255.0

            visualization = show_cam_on_image(
                image_for_viz, grayscale_cam, use_rgb=True
            )

            # Save visualization
            image_name = os.path.basename(image_path)
            name, _ = os.path.splitext(image_name)

            viz_path = os.path.join(
                output_folder, f"{name}_efficientnet_explainability.jpg"
            )
            face_path = os.path.join(output_folder, f"{name}_efficientnet_face.jpg")

            cv2.imwrite(viz_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            cv2.imwrite(
                face_path,
                cv2.cvtColor((image_for_viz * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            )

        except Exception as e:
            print(f"Error generating explainability visualization: {e}")

    def predict_single_image(self, image_path: str) -> tuple:
        """Predicts on a single image and returns raw scores (compatible with notebook format).

        Args:
            image_path: Path to the input image

        Returns:
            tuple: (is_real_score, is_fake_score) where scores are between 0 and 1
        """
        # Load and preprocess image
        image = self._load_image(image_path)
        input_tensor = self._preprocess_image(image)

        # Make prediction
        with torch.no_grad():
            output = self.model(input_tensor)

            # Handle different output formats
            if isinstance(output, tuple):
                logits = output[1] if len(output) > 1 else output[0]
            else:
                logits = output

            # Apply sigmoid and get scores
            if logits.shape[-1] == 1:
                # Single output - treat as fake probability
                fake_score = torch.sigmoid(logits).item()
                real_score = 1 - fake_score
            else:
                # Two outputs - softmax
                probs = torch.softmax(logits, dim=1)
                real_score = probs[0, 0].item()
                fake_score = probs[0, 1].item()

        return real_score, fake_score
