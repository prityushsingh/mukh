"""Deepfake detection pipeline for running multiple models and combining results using weighted aggregation."""

import csv
import os
from typing import Callable, Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..core.types import DeepfakeDetection
from ..deepfake_detection import DeepfakeDetector


class DeepfakeDetectionPipeline:
    """Pipeline for running multiple deepfake detection models and combining results using weighted aggregation.

    This class allows you to run multiple deepfake detection models on the same
    input and combine their predictions using weighted averaging.

    Attributes:
        models: Dictionary mapping model names to DeepfakeDetector instances
        model_weights: Dictionary mapping model names to their weights for weighted averaging
        confidence_threshold: Final confidence threshold for pipeline decisions
    """

    def __init__(
        self,
        model_configs: List[Dict],
        model_weights: Dict[str, float],
        confidence_threshold: float = 0.5,
    ):
        """Initializes the deepfake detection pipeline.

        Args:
            model_configs: List of dictionaries containing model configuration.
                Each dict should have 'name' and optionally 'model_path', 'device', etc.
                Example: [
                    {'name': 'resnet_inception', 'confidence_threshold': 0.4},
                    {'name': 'resnext', 'model_variant': 'resnext'},
                    {'name': 'efficientnet', 'net_model': 'EfficientNetB4'}
                ]
            model_weights: Dictionary mapping model names to weights for weighted averaging
            confidence_threshold: Final confidence threshold for pipeline decisions

        Raises:
            ValueError: If invalid model configuration provided
        """
        self.models = {}
        self.model_weights = model_weights
        self.confidence_threshold = confidence_threshold

        # Initialize models
        self._initialize_models(model_configs)

        # Validate and normalize model weights
        self._validate_model_weights()

    def _initialize_models(self, model_configs: List[Dict]) -> None:
        """Initializes deepfake detection models from configurations.

        Args:
            model_configs: List of model configuration dictionaries

        Raises:
            ValueError: If model configuration is invalid
        """
        for config in model_configs:
            if "name" not in config:
                raise ValueError("Each model config must have a 'name' field")

            model_name = config["name"]

            # Extract model-specific parameters
            model_params = config.copy()
            del model_params["name"]

            # Create detector instance
            try:
                detector = DeepfakeDetector(model_name=model_name, **model_params)
                self.models[model_name] = detector
                print(f"Initialized model: {model_name}")
            except Exception as e:
                raise ValueError(f"Failed to initialize model '{model_name}': {e}")

    def _validate_model_weights(self) -> None:
        """Validates that model weights are provided for all models and normalizes them.

        Raises:
            ValueError: If weights are missing or invalid
        """
        model_names = set(self.models.keys())
        weight_names = set(self.model_weights.keys())

        if model_names != weight_names:
            missing_weights = model_names - weight_names
            extra_weights = weight_names - model_names

            error_msg = []
            if missing_weights:
                error_msg.append(f"Missing weights for models: {missing_weights}")
            if extra_weights:
                error_msg.append(f"Extra weights for unknown models: {extra_weights}")

            raise ValueError("; ".join(error_msg))

        # Validate weight values
        for name, weight in self.model_weights.items():
            if not isinstance(weight, (int, float)) or weight < 0:
                raise ValueError(
                    f"Weight for model '{name}' must be a non-negative number"
                )

        # Normalize weights to sum to 1
        total_weight = sum(self.model_weights.values())
        if total_weight == 0:
            raise ValueError("Sum of model weights cannot be zero")

        self.model_weights = {
            name: weight / total_weight for name, weight in self.model_weights.items()
        }

    def _aggregate_predictions(
        self, predictions: List[DeepfakeDetection], frame_number: int = 0
    ) -> DeepfakeDetection:
        """Aggregates predictions from multiple models using weighted averaging.

        Args:
            predictions: List of predictions from different models
            frame_number: Frame number for the aggregated result

        Returns:
            DeepfakeDetection: Weighted averaged prediction result
        """
        if not predictions:
            raise ValueError("Cannot aggregate empty predictions list")

        weighted_deepfake_prob = 0.0
        total_weight = 0.0
        model_names = []

        for pred in predictions:
            # Extract base model name (remove any suffixes)
            base_model_name = pred.model_name.split("-")[0].lower()

            # Map model names to the keys used in model_weights
            if base_model_name == "resnetinception" or base_model_name.startswith(
                "resnet"
            ):
                base_model_name = "resnet_inception"
            elif base_model_name == "resnext":
                base_model_name = "resnext"
            elif base_model_name == "efficientnet":
                base_model_name = "efficientnet"

            weight = self.model_weights.get(base_model_name, 0.0)

            # Convert confidence to deepfake probability
            # Each model returns confidence differently, so we need to normalize
            if pred.is_deepfake:
                deepfake_prob = pred.confidence
            else:
                deepfake_prob = 1.0 - pred.confidence

            weighted_deepfake_prob += deepfake_prob * weight
            total_weight += weight
            model_names.append(pred.model_name)

        # Normalize by total weight (should be 1.0 if weights are normalized)
        if total_weight > 0:
            weighted_deepfake_prob /= total_weight

        is_deepfake = weighted_deepfake_prob >= self.confidence_threshold
        final_confidence = (
            weighted_deepfake_prob if is_deepfake else (1.0 - weighted_deepfake_prob)
        )
        combined_model_name = f"WeightedPipeline({'+'.join(model_names)})"

        return DeepfakeDetection(
            frame_number=frame_number,
            is_deepfake=is_deepfake,
            confidence=float(final_confidence),
            model_name=combined_model_name,
        )

    def detect_image(
        self,
        image_path: str,
        save_csv: bool = False,
        csv_path: str = "output/pipeline_deepfake_detections.csv",
        save_annotated: bool = False,
        output_folder: str = "output",
        save_individual_results: bool = False,
    ) -> DeepfakeDetection:
        """Detects deepfake in the given image using all models in the pipeline.

        Args:
            image_path: Path to the input image
            save_csv: Whether to save detection results to CSV file
            csv_path: Path where to save the CSV file
            save_annotated: Whether to save annotated image with results
            output_folder: Folder path where to save annotated images
            save_individual_results: Whether to save individual model results

        Returns:
            DeepfakeDetection: Aggregated detection result
        """
        predictions = []

        # Run detection with each model
        for model_name, detector in self.models.items():
            try:
                print(f"Running detection with model: {model_name}")
                prediction = detector.detect_image(
                    image_path=image_path,
                    save_csv=save_individual_results and save_csv,
                    csv_path=(
                        f"{model_name}_{csv_path}"
                        if save_individual_results
                        else csv_path
                    ),
                    save_annotated=save_individual_results and save_annotated,
                    output_folder=(
                        os.path.join(output_folder, model_name)
                        if save_individual_results
                        else output_folder
                    ),
                )
                predictions.append(prediction)
                print(f"Prediction completed for {model_name}")
            except Exception as e:
                print(f"Warning: Model {model_name} failed on image {image_path}: {e}")
                continue

        if not predictions:
            raise RuntimeError("All models failed to process the image")

        # Aggregate predictions using weighted averaging
        aggregated_result = self._aggregate_predictions(predictions, frame_number=0)

        # Print summary
        self._print_image_summary(image_path, predictions, aggregated_result)

        # Save aggregated results
        if save_csv:
            self._save_aggregated_results([aggregated_result], image_path, csv_path)

        if save_annotated:
            self._save_annotated_result(image_path, aggregated_result, output_folder)

        return aggregated_result

    def detect_video(
        self,
        video_path: str,
        save_csv: bool = False,
        csv_path: str = "output/pipeline_deepfake_detections.csv",
        save_annotated: bool = False,
        output_folder: str = "output",
        num_frames: int = 11,
        save_individual_results: bool = False,
    ) -> List[DeepfakeDetection]:
        """Detects deepfake in the given video using all models in the pipeline.

        Args:
            video_path: Path to the input video
            save_csv: Whether to save detection results to CSV file
            csv_path: Path where to save the CSV file
            save_annotated: Whether to save annotated video with results
            output_folder: Folder path where to save annotated videos
            num_frames: Number of equally spaced frames to analyze
            save_individual_results: Whether to save individual model results

        Returns:
            List[DeepfakeDetection]: List of aggregated detection results for each frame
        """
        all_model_predictions = {}

        # Run detection with each model
        for model_name, detector in self.models.items():
            try:
                print(f"Running detection with model: {model_name}")
                predictions = detector.detect_video(
                    video_path=video_path,
                    save_csv=save_individual_results and save_csv,
                    csv_path=(
                        f"{model_name}_{csv_path}"
                        if save_individual_results
                        else csv_path
                    ),
                    save_annotated=save_individual_results and save_annotated,
                    output_folder=(
                        os.path.join(output_folder, model_name)
                        if save_individual_results
                        else output_folder
                    ),
                    num_frames=num_frames,
                )
                all_model_predictions[model_name] = {
                    pred.frame_number: pred for pred in predictions
                }
                print(f"Predictions completed for {model_name}")
            except Exception as e:
                print(f"Warning: Model {model_name} failed on video {video_path}: {e}")
                continue

        if not all_model_predictions:
            raise RuntimeError("All models failed to process the video")

        # Get all frame numbers that were processed
        all_frame_numbers = set()
        for predictions in all_model_predictions.values():
            all_frame_numbers.update(predictions.keys())

        # Aggregate predictions for each frame using weighted averaging
        aggregated_results = []
        for frame_number in sorted(all_frame_numbers):
            frame_predictions = []

            for model_name, predictions in all_model_predictions.items():
                if frame_number in predictions:
                    frame_predictions.append(predictions[frame_number])

            if frame_predictions:
                aggregated_result = self._aggregate_predictions(
                    frame_predictions, frame_number
                )
                aggregated_results.append(aggregated_result)

        # Print summary
        self._print_video_summary(video_path, all_model_predictions, aggregated_results)

        # Save aggregated results
        if save_csv and aggregated_results:
            self._save_aggregated_results(aggregated_results, video_path, csv_path)

        if save_annotated and aggregated_results:
            self._save_annotated_video_result(
                video_path, aggregated_results, output_folder
            )

        return aggregated_results

    def detect(
        self,
        media_path: str,
        save_csv: bool = False,
        csv_path: str = "output/pipeline_deepfake_detections.csv",
        save_annotated: bool = False,
        output_folder: str = "output",
        num_frames: int = 11,
        save_individual_results: bool = False,
    ) -> Union[DeepfakeDetection, List[DeepfakeDetection]]:
        """Detects deepfake in the given media file using all models in the pipeline.

        Automatically determines whether the input is an image or video
        based on file extension and calls the appropriate detection method.

        Args:
            media_path: Path to the input media file (image or video)
            save_csv: Whether to save detection results to CSV file
            csv_path: Path where to save the CSV file
            save_annotated: Whether to save annotated media with results
            output_folder: Folder path where to save annotated media
            num_frames: Number of equally spaced frames to analyze for videos
            save_individual_results: Whether to save individual model results

        Returns:
            DeepfakeDetection for images, List[DeepfakeDetection] for videos

        Raises:
            ValueError: If the file format is not supported
            FileNotFoundError: If the media file doesn't exist
        """
        if not os.path.exists(media_path):
            raise FileNotFoundError(f"Media file not found: {media_path}")

        # Get file extension
        _, ext = os.path.splitext(media_path.lower())

        # Image extensions
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        # Video extensions
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

        if ext in image_extensions:
            return self.detect_image(
                image_path=media_path,
                save_csv=save_csv,
                csv_path=csv_path,
                save_annotated=save_annotated,
                output_folder=output_folder,
                save_individual_results=save_individual_results,
            )
        elif ext in video_extensions:
            return self.detect_video(
                video_path=media_path,
                save_csv=save_csv,
                csv_path=csv_path,
                save_annotated=save_annotated,
                output_folder=output_folder,
                num_frames=num_frames,
                save_individual_results=save_individual_results,
            )
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _save_aggregated_results(
        self,
        results: List[DeepfakeDetection],
        media_path: str,
        csv_path: str = "output/pipeline_deepfake_detections.csv",
    ) -> None:
        """Saves aggregated pipeline results to CSV file.

        Args:
            results: List of aggregated detection results
            media_path: Path to the source media file
            csv_path: Path where to save the CSV file
        """
        # Create directory only if csv_path contains a directory component
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "media_path",
                "frame_number",
                "is_deepfake",
                "confidence",
                "model_name",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header if file is new
            if not file_exists:
                writer.writeheader()

            # Write detection results
            for result in results:
                writer.writerow(
                    {
                        "media_path": media_path,
                        "frame_number": result.frame_number,
                        "is_deepfake": result.is_deepfake,
                        "confidence": result.confidence,
                        "model_name": result.model_name,
                    }
                )

    def _save_annotated_result(
        self, image_path: str, result: DeepfakeDetection, output_folder: str
    ) -> None:
        """Saves annotated image with pipeline result.

        Args:
            image_path: Path to the original image
            result: Aggregated detection result
            output_folder: Directory to save annotated image
        """
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return

        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Create drawing context
        draw = ImageDraw.Draw(pil_image)

        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            except:
                font = ImageFont.load_default()

        # Prepare text
        status = "DEEPFAKE" if result.is_deepfake else "REAL"
        confidence_text = f"{result.confidence:.3f}"
        text = f"{status} ({confidence_text})"

        # Choose color based on result
        color = (
            (255, 0, 0) if result.is_deepfake else (0, 255, 0)
        )  # Red for deepfake, green for real

        # Get text size
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position text at top-left with some padding
        x, y = 10, 10

        # Draw background rectangle for text
        draw.rectangle(
            [x - 5, y - 5, x + text_width + 5, y + text_height + 5], fill=(0, 0, 0, 128)
        )

        # Draw text
        draw.text((x, y), text, fill=color, font=font)

        # Save annotated image
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{name}_pipeline_annotated{ext}")

        # Convert back to BGR for OpenCV saving
        annotated_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, annotated_image)

        print(f"Saved annotated image: {output_path}")

    def _save_annotated_video_result(
        self, video_path: str, results: List[DeepfakeDetection], output_folder: str
    ) -> None:
        """Saves annotated video with pipeline results.

        Args:
            video_path: Path to the original video
            results: List of aggregated detection results
            output_folder: Directory to save annotated video
        """
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)

        # Open input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Prepare output video
        filename = os.path.basename(video_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{name}_pipeline_annotated.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Create a mapping of frame numbers to results
        results_dict = {result.frame_number: result for result in results}

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Check if we have a detection result for this frame
            if frame_idx in results_dict:
                result = results_dict[frame_idx]

                # Prepare text
                status = "DEEPFAKE" if result.is_deepfake else "REAL"
                confidence_text = f"{result.confidence:.3f}"
                text = f"{status} ({confidence_text})"

                # Choose color based on result
                color = (
                    (0, 0, 255) if result.is_deepfake else (0, 255, 0)
                )  # Red for deepfake, green for real

                # Add text to frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2

                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, font, font_scale, thickness
                )

                # Position text at top-left with some padding
                x, y = 10, 30

                # Draw background rectangle for text
                cv2.rectangle(
                    frame,
                    (x - 5, y - text_height - 5),
                    (x + text_width + 5, y + baseline + 5),
                    (0, 0, 0),
                    -1,
                )

                # Draw text
                cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

            # Write frame to output video
            out.write(frame)
            frame_idx += 1

        # Release resources
        cap.release()
        out.release()

        print(f"Saved annotated video: {output_path}")

    def get_pipeline_info(self) -> Dict:
        """Returns information about the pipeline configuration.

        Returns:
            Dictionary containing pipeline information including models and configuration details.
        """
        model_info = {}
        for name, detector in self.models.items():
            model_info[name] = detector.get_model_info()

        return {
            "models": model_info,
            "aggregation_strategy": "weighted_average",
            "model_weights": self.model_weights,
            "confidence_threshold": self.confidence_threshold,
            "num_models": len(self.models),
        }

    def set_confidence_threshold(self, threshold: float) -> None:
        """Updates the confidence threshold for pipeline decisions.

        Args:
            threshold: New confidence threshold (0.0 to 1.0)

        Raises:
            ValueError: If threshold is not between 0.0 and 1.0
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

        self.confidence_threshold = threshold

    def _print_image_summary(
        self,
        image_path: str,
        predictions: List[DeepfakeDetection],
        aggregated_result: DeepfakeDetection,
    ) -> None:
        """Prints a summary of image detection results.

        Args:
            image_path: Path to the processed image
            predictions: List of individual model predictions
            aggregated_result: Final aggregated result
        """
        print("\n" + "=" * 80)
        print(f"PIPELINE SUMMARY - IMAGE: {os.path.basename(image_path)}")
        print("=" * 80)

        print("\nIndividual Model Results:")
        print("-" * 50)
        for pred in predictions:
            status = "DEEPFAKE" if pred.is_deepfake else "REAL"
            print(
                f"{pred.model_name:20} | Confidence: {pred.confidence:.4f} | Verdict: {status}"
            )

        print("\nWeighted Aggregation:")
        print("-" * 50)
        for model_name, weight in self.model_weights.items():
            print(f"{model_name:20} | Weight: {weight:.4f}")

        print(f"\nFINAL VERDICT:")
        print("-" * 50)
        final_status = "DEEPFAKE" if aggregated_result.is_deepfake else "REAL"
        print(f"Frame Number: {aggregated_result.frame_number}")
        print(f"Final Confidence: {aggregated_result.confidence:.4f}")
        print(f"Final Verdict: {final_status}")
        print(f"Threshold Used: {self.confidence_threshold}")
        print("=" * 80 + "\n")

    def _print_video_summary(
        self,
        video_path: str,
        all_model_predictions: Dict,
        aggregated_results: List[DeepfakeDetection],
    ) -> None:
        """Prints a summary of video detection results.

        Args:
            video_path: Path to the processed video
            all_model_predictions: Dictionary of all model predictions by frame
            aggregated_results: List of final aggregated results
        """
        print("\n" + "=" * 100)
        print(f"PIPELINE SUMMARY - VIDEO: {os.path.basename(video_path)}")
        print("=" * 100)

        print(f"\nProcessed {len(aggregated_results)} frames")
        print(f"Models used: {list(all_model_predictions.keys())}")

        print("\nModel Weights:")
        print("-" * 50)
        for model_name, weight in self.model_weights.items():
            print(f"{model_name:20} | Weight: {weight:.4f}")

        print(f"\nFrame-by-Frame Results:")
        print("-" * 100)
        print(f"{'Frame':<8} | ", end="")

        # Print model headers
        model_names = list(all_model_predictions.keys())
        for model_name in model_names:
            print(f"{model_name[:12]:<14} | ", end="")
        print(f"{'Final Conf':<12} | {'Final Verdict':<12}")

        print("-" * 100)

        # Print results for each frame
        for result in aggregated_results:
            frame_num = result.frame_number
            print(f"{frame_num:<8} | ", end="")

            # Print individual model confidences for this frame
            for model_name in model_names:
                if frame_num in all_model_predictions[model_name]:
                    conf = all_model_predictions[model_name][frame_num].confidence
                    print(f"{conf:<14.4f} | ", end="")
                else:
                    print(f"{'N/A':<14} | ", end="")

            # Print final result
            final_status = "DEEPFAKE" if result.is_deepfake else "REAL"
            print(f"{result.confidence:<12.4f} | {final_status:<12}")

        # Calculate overall statistics
        deepfake_frames = sum(1 for r in aggregated_results if r.is_deepfake)
        real_frames = len(aggregated_results) - deepfake_frames
        avg_confidence = sum(r.confidence for r in aggregated_results) / len(
            aggregated_results
        )

        print("\nOverall Statistics:")
        print("-" * 50)
        print(f"Total Frames Analyzed: {len(aggregated_results)}")
        print(
            f"Frames Classified as DEEPFAKE: {deepfake_frames} ({deepfake_frames/len(aggregated_results)*100:.1f}%)"
        )
        print(
            f"Frames Classified as REAL: {real_frames} ({real_frames/len(aggregated_results)*100:.1f}%)"
        )
        print(f"Average Confidence: {avg_confidence:.4f}")
        print(f"Threshold Used: {self.confidence_threshold}")

        # Overall video verdict based on majority of frames
        overall_verdict = "DEEPFAKE" if deepfake_frames > real_frames else "REAL"
        print(f"\nOVERALL VIDEO VERDICT: {overall_verdict}")
        print("=" * 100 + "\n")
